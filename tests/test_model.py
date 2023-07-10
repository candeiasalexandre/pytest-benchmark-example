from typing import Dict, List
from pytest_benchmarks_example.model import (
    EmbeddingConfig,
    NamedTensor,
    SimpleRegressionModelData,
    SimpleRegressionModel,
)
import torch
import pytest


def generate_data(
    categorical_data_info: List[EmbeddingConfig],
    numerical_data_info: List[str],
    n_rows: int,
) -> SimpleRegressionModelData:

    _categorical_data = []
    _categorical_data_columns = []
    for embedding_cfg in categorical_data_info:
        _categorical_data.append(
            torch.randint(0, embedding_cfg.cardinality - 1, (n_rows, 1))
        )
        _categorical_data_columns.append(embedding_cfg.name)

    _numerical_data = torch.randn((n_rows, len(numerical_data_info)))

    return SimpleRegressionModelData(
        numericals=NamedTensor(tuple(numerical_data_info), _numerical_data),
        categoricals=NamedTensor(
            tuple(_categorical_data_columns), torch.cat(_categorical_data, dim=1)
        ),
    )


@pytest.fixture
def categorical_data_info() -> List[EmbeddingConfig]:
    return [
        EmbeddingConfig("feature_a", 10, 2),
        EmbeddingConfig("feature_b", 100, 10),
        EmbeddingConfig("feature_c", 123, 8),
        EmbeddingConfig("feature_d", 250, 12),
    ]


@pytest.fixture
def numerical_data_info() -> List[str]:
    return ["feature_e", "feature_f"]


def create_dummy_model(
    categorical_data_info: List[EmbeddingConfig],
    numerical_data_info: List[str],
    hidden_layers: List[int],
) -> SimpleRegressionModel:
    return SimpleRegressionModel(
        embeddings=categorical_data_info,
        hidden_layers=hidden_layers,
        numerical_cols=numerical_data_info,
    )


@pytest.mark.parametrize(
    "n_hidden_layers",
    [
        pytest.param(8, id="n_hidden_layers=8"),
        pytest.param(4, id="n_hidden_layers=4"),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_benchmark_model(
    benchmark,
    batch_size: int,
    n_hidden_layers: int,
    categorical_data_info: List[EmbeddingConfig],
    numerical_data_info: List[str],
) -> None:

    n_rows = batch_size
    data = generate_data(categorical_data_info, numerical_data_info, n_rows)

    hidden_layers = [2**n for n in range(1, n_hidden_layers)]
    hidden_layers.reverse()
    dummy_model = create_dummy_model(
        categorical_data_info, numerical_data_info, hidden_layers
    )

    dummy_model.eval()
    with torch.no_grad():
        benchmark(dummy_model.forward, data)
