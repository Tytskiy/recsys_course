from dataclasses import dataclass, field

from engine.interfaces import ID, Factor


@dataclass(frozen=True)
class TrainingArtifacts:
    files: list[str] = field(default_factory=list)
    item_profiles: dict[ID, dict[str, Factor]] = field(default_factory=dict)
