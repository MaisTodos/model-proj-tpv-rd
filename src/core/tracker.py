import os
import mlflow
from typing import Any, Dict, List, Optional
from src.core.telemetria import get_logger

logger = get_logger("mlops.tracking")

# Tipos internos do scikit-learn que o skops (serializador padrão do mlflow.sklearn)
# marca como "não confiáveis" por padrão. Cobre as árvores de decisão/ensemble
# (RandomForest, GradientBoosting, ExtraTrees) usadas pelo TemplateModelTrainer.
_DEFAULT_TRUSTED_SKOPS_TYPES = ["sklearn.tree._tree.Tree"]

# Fallback local (dev) de tracking URI, compartilhado com src/config/predict.py.
DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"

class TrackerClient:
    """
    Facade para o MLflow. Isola o código do cientista da infraestrutura de tracking.
    """
    def __init__(self, experiment_name: str):
        # Fallback local (dev): backend de arquivo puro está em modo de manutenção no MLflow
        # 3.x — usar SQLite como vem recomendado em https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store.
        # Em produção, MLFLOW_TRACKING_URI deve apontar para o servidor MLflow (ADR-004).
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        # Bucket de artefatos (ADR-004): S3_URI é o nome definido no .env do projeto;
        # MLFLOW_ARTIFACT_URI fica como alias secundário por compatibilidade.
        self.artifact_uri = os.getenv("S3_URI") or os.getenv("MLFLOW_ARTIFACT_URI", "s3://default-mais-todos-artifacts/")
        
        mlflow.set_tracking_uri(self.tracking_uri)
        # mlflow.set_experiment() não aceita artifact_location: o experimento precisa ser
        # criado explicitamente (uma única vez) para fixar onde os artefatos serão gravados.
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, artifact_location=self.artifact_uri)
        self.experiment = mlflow.set_experiment(experiment_name)
        logger.info("Tracker configurado", experiment=experiment_name, tracking_uri=self.tracking_uri)

    def start_run(self, run_name: Optional[str] = None):
        """Inicia a run no MLflow. Deve ser usado com context manager (with)."""
        return mlflow.start_run(run_name=run_name)

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float) -> None:
        mlflow.log_metric(key, value)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str = "model", trusted_types: Optional[List[str]] = None) -> None:
        """
        Registra o modelo no MLflow (flavor scikit-learn).
        O armazenamento físico irá para o bucket S3 (Mais Todos) configurado na inicialização.

        `trusted_types`: tipos adicionais a confiar na desserialização via skops, caso o
        algoritmo escolhido pelo Cientista de Dados acuse outros tipos internos como não
        confiáveis (ver exceção do mlflow.sklearn.log_model).
        """
        # A própria classe do objeto é código do Cientista de Dados (ex: TemplatePreprocessor)
        # e é, por definição, confiável para quem está logando-a.
        own_type = f"{type(model).__module__}.{type(model).__qualname__}"
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            skops_trusted_types=_DEFAULT_TRUSTED_SKOPS_TYPES + [own_type] + (trusted_types or []),
        )
        logger.info("Modelo logado no tracking", artifact_path=artifact_path)

    def register_model(self, run_id: str, model_name: str) -> None:
        """
        Regista o modelo no Model Registry (ADR-004).
        """
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info("Modelo promovido ao Registry", model_name=model_name, model_uri=model_uri)