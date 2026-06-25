"""
ModelManager com warm swap para pipeline lateral → top-view → lateral → ...

Estratégia:
- Ambos os modelos são carregados do disco UMA VEZ no início (warm-up).
- Apenas o modelo ativo fica na GPU (VRAM).
- O modelo inativo tem seus parâmetros movidos para CPU RAM.
- Trocar de modelo = mover pesos CPU→GPU e GPU→CPU. Custo ~100-300ms
  vs ~2-5s de reload do disco na Jetson Nano.
- A primeira inferência de cada modelo ainda é feita no warm-up,
  eliminando o custo de JIT/compilação no pipeline real.
"""

import gc
import torch
import numpy as np
from rfdetr import RFDETRSegMedium


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models: dict[str, RFDETRSegMedium] = {}
        self._active: str | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = True

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def warmup(self, model_paths: list[str], resolution: int = 1296,
               dummy_size: tuple[int, int] = (640, 640)):
        """
        Carrega todos os modelos do disco e faz uma inferência dummy
        em cada um para eliminar overhead de JIT/compilação.

        Chame isso UMA VEZ antes do loop principal.

        Parâmetros
        ----------
        model_paths : list[str]
            Caminhos dos pesos, na ordem em que serão usados primeiro.
            O primeiro da lista fica na GPU ao terminar o warmup.
        resolution : int
            Resolução de entrada do RF-DETR.
        dummy_size : tuple
            Tamanho da imagem dummy para o warm-up de JIT.
        """
        from PIL import Image

        print("[ModelManager] Iniciando warm-up...")

        for path in model_paths:
            print(f"  Carregando: {path}")
            model = RFDETRSegMedium(pretrain_weights=path, resolution=resolution)
            model.optimize_for_inference()
            self._models[path] = model

        # Inferência dummy em cada modelo para compilar kernels CUDA
        dummy = Image.fromarray(
            np.zeros((dummy_size[1], dummy_size[0], 3), dtype=np.uint8)
        )

        for path in model_paths:
            print(f"  Warm-up JIT: {path}")
            self._move_to_gpu(path)
            try:
                self._models[path].predict(dummy, threshold=0.5)
            except Exception:
                pass  # resultado dummy pode ser inválido, tudo bem

        # Deixa o primeiro ativo na GPU, move os demais para CPU
        self._move_to_gpu(model_paths[0])
        for path in model_paths[1:]:
            self._move_to_cpu(path)

        print(f"[ModelManager] Warm-up concluído. Ativo na GPU: {model_paths[0]}")

    def use(self, model_path: str):
        """
        Torna o modelo especificado o ativo na GPU.
        Se já estiver ativo, não faz nada (sem custo).
        """
        if model_path not in self._models:
            raise KeyError(
                f"Modelo '{model_path}' não foi carregado no warm-up. "
                f"Disponíveis: {list(self._models.keys())}"
            )

        if self._active == model_path:
            return

        if self._active is not None:
            self._move_to_cpu(self._active)

        self._move_to_gpu(model_path)

    def predict(self, pil_img, threshold: float = 0.5):
        if self._active is None:
            raise RuntimeError("Nenhum modelo ativo. Chame warmup() e use() primeiro.")
        return self._models[self._active].predict(pil_img, threshold=threshold)

    def shutdown(self):
        """Libera todos os modelos. Chame ao fim da aplicação."""
        print("[ModelManager] Desligando...")
        for model in self._models.values():
            del model
        self._models.clear()
        self._active = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _move_to_gpu(self, path: str):
        self._try_move(self._models[path], self._device)
        self._active = path
        print(f"[ModelManager] → GPU: {path}")

    def _move_to_cpu(self, path: str):
        self._try_move(self._models[path], torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[ModelManager] → CPU: {path}")

    @staticmethod
    def _try_move(model, device):
        """Move os pesos do modelo para o device. Tenta atributos comuns do rfdetr."""
        moved = False
        for attr in ("model", "net", "backbone", "detector"):
            sub = getattr(model, attr, None)
            if sub is not None and isinstance(sub, torch.nn.Module):
                sub.to(device)
                moved = True
                break
        if not moved:
            # Fallback: tenta o próprio objeto
            if isinstance(model, torch.nn.Module):
                model.to(device)


# Instância global — importe nos outros módulos
manager = ModelManager()