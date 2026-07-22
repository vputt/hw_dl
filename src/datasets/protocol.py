from dataclasses import dataclass
from pathlib import Path


LABEL_TO_INDEX = {"spoof": 0, "bonafide": 1}


@dataclass(frozen=True)
class ProtocolEntry:
    speaker_id: str
    key: str
    environment_id: str
    attack_id: str
    label: str

    @property
    def label_index(self) -> int:
        return LABEL_TO_INDEX[self.label]


def read_protocol(path: str | Path) -> list[ProtocolEntry]:
    """Читает официальный ASVspoof 2019 LA"""
    entries: list[ProtocolEntry] = []
    with Path(path).open("r", encoding="utf-8") as protocol_file:
        for raw_line in protocol_file:
            line = raw_line.strip()
            if not line:
                continue
            entries.append(ProtocolEntry(*line.split()))
    return entries
