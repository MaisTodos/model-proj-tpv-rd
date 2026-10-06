"""
Entrypoint central de MLOps.
Roteia a execução das etapas de pipeline batch via CLI com base no project.yml.
"""
import argparse
import sys
import yaml
import importlib

# Garante que a observabilidade (Datadog) seja inicializada primeiro (ADR-001)
from src.core.telemetria import get_logger

logger = get_logger("mlops.main")

def load_config(config_path: str = "project.yml") -> dict:
    """Lê as definições centrais do projeto."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Arquivo project.yml não encontrado.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Entrypoint do Pipeline MLOps Batch")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["feature_engineering", "training", "inference"],
        help="Define qual etapa do pipeline será executada."
    )
    args = parser.parse_args()
    
    config = load_config()
    step = args.step

    logger.info("Iniciando execução via main.py", step=step, project=config.get("name"))

    pipelines = config.get("pipelines", {})
    if step not in pipelines:
        logger.error("Etapa não mapeada no bloco 'pipelines' do project.yml", step=step)
        sys.exit(1)

    # Converte o caminho (ex: 'src/config/train.py') para notação de módulo (ex: 'src.config.train')
    module_path = pipelines[step]
    module_name = module_path.replace("/", ".").replace(".py", "")

    try:
        logger.info("Carregando módulo de execução", module=module_name)
        step_module = importlib.import_module(module_name)
        
        if hasattr(step_module, 'main'):
            # Aciona a função main() do script específico da etapa
            step_module.main()
        else:
            logger.error(f"O módulo {module_name} não possui uma função main().")
            sys.exit(1)
            
    except Exception as e:
        logger.error("Falha fatal na execução da pipeline", step=step, error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()