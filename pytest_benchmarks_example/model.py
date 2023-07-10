from typing import Dict, List, Optional, Tuple, Type
import torch
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    name: str
    cardinality: int
    dim: int


@dataclass
class NamedTensor:
    columns: Tuple[str]
    data: torch.Tensor
    _column_idx_map: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self._validate_data()
        self._column_idx_map = {column: idx for idx, column in enumerate(self.columns)}

    def _validate_data(self):
        if len(self.data.shape) != 2:
            raise RuntimeError("NamedTensor only supports data of dim=2!")
        if len(self.columns) != self.data.shape[1]:
            raise RuntimeError(
                "Number of columns should be the same as number size of dim=2!"
            )

    def get_data(self, columns: List[str]) -> torch.Tensor:
        idx_columns = [self._column_idx_map[column] for column in columns]
        return self.data[:, idx_columns]

    def get_data_at_idx(self, idx: int) -> "NamedTensor":
        return NamedTensor(self.columns, self.data[idx, :])


@dataclass
class SimpleRegressionModelData:
    numericals: NamedTensor
    categoricals: NamedTensor


class SimpleRegressionModel(torch.nn.Module):
    def __init__(
        self,
        embeddings: List[EmbeddingConfig],
        hidden_layers: List[int],
        numerical_cols: List[str],
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self._numerical_cols = numerical_cols.copy()
        self._categorical_cols = list(map(lambda x: x.name, embeddings))

        self._embeddings = torch.nn.ModuleDict(
            {x.name: torch.nn.Embedding(x.cardinality, x.dim) for x in embeddings}
        )

        all_embedding_dim = sum(map(lambda x: x.dim, embeddings))
        first_hidden_state_size = all_embedding_dim + len(numerical_cols)

        layers = self._create_mlp_layers(
            first_hidden_state_size, hidden_layers, dropout
        )
        self._mlp = torch.nn.Sequential(*layers)

    def _create_mlp_layers(
        self, input_size: int, hidden_layers: List[int], dropout: Optional[float] = None
    ) -> List[torch.nn.Module]:
        _layers = []
        prev_layer_size = input_size
        for layer_size in hidden_layers:
            _layers.append(torch.nn.Linear(prev_layer_size, layer_size))
            if dropout is not None:
                _layers.append(torch.nn.Dropout(dropout))
            _layers.append(self.activation())
            prev_layer_size = layer_size

        _layers.append(torch.nn.Linear(prev_layer_size, 1))
        _layers.append(self.output_activation())

        return _layers

    def forward(self, x: SimpleRegressionModelData) -> torch.FloatTensor:

        categoricals = x.categoricals

        embeddings_by_column = [
            self._embeddings[column](categoricals.get_data([column])).squeeze(dim=1)
            for column in self._categorical_cols
        ]
        embeddings_batch = torch.concat(
            embeddings_by_column,
            dim=1,
        )  # this is # batch_size x (sum embedding_dim)

        numericals = x.numericals.get_data(self._numerical_cols)

        numericals_and_embeddings_batch = torch.concat(
            [embeddings_batch, numericals], dim=1
        )

        return self._mlp(numericals_and_embeddings_batch)

    @property
    def activation(self) -> Type[torch.nn.Module]:
        return torch.nn.ReLU

    @property
    def output_activation(self) -> Type[torch.nn.Module]:
        return torch.nn.ReLU
