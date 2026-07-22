import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    "train/loss",
    "dev/loss",
    "eval/loss",
    "dev/eer_percent",
    "eval/eer_percent",
    "dev/eer_threshold",
    "learning_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Графики для отчёта из W&B CSV")
    parser.add_argument("--history", nargs="+", required=True, help="Один или несколько CSV")
    parser.add_argument("--output-dir", default="report_figures")
    parser.add_argument("--per-attack", default=None, help="CSV с EER по атакам")
    return parser.parse_args()


def find_column(columns: list[str], metric: str) -> str | None:
    for column in columns:
        if column.endswith(("__MIN", "__MAX")):
            continue
        if column == metric or column.endswith(f" - {metric}"):
            return column
    return None


def read_history(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path_value in paths:
        frame = pd.read_csv(path_value)
        epoch_column = find_column(frame.columns.tolist(), "epoch")
        if epoch_column is None:
            continue

        clean = pd.DataFrame(
            {"epoch": pd.to_numeric(frame[epoch_column], errors="coerce")}
        )
        for metric in METRICS:
            column = find_column(frame.columns.tolist(), metric)
            if column is not None:
                clean[metric] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(clean.dropna(subset=["epoch"]))

    if not frames:
        raise ValueError("В CSV не найден столбец epoch")

    history = pd.concat([frame.set_index("epoch") for frame in frames])
    history = history.groupby(level=0).first().sort_index().reset_index()
    history["epoch"] = history["epoch"].astype(int)
    return history


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_losses(history: pd.DataFrame, output_dir: Path) -> None:
    required = {"train/loss", "dev/loss"}
    if not required.issubset(history.columns):
        raise ValueError(f"Не найдены метрики: {sorted(required - set(history.columns))}")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history["epoch"], history["train/loss"], marker="o", label="Train CE")
    axis.plot(history["epoch"], history["dev/loss"], marker="o", label="Dev CE")
    if "eval/loss" in history:
        axis.plot(history["epoch"], history["eval/loss"], marker="o", label="Eval CE")
    axis.set(title="Cross-Entropy по эпохам", xlabel="Эпоха", ylabel="Cross-Entropy")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output_dir, "loss_by_epoch")


def plot_eer(history: pd.DataFrame, output_dir: Path) -> None:
    required = {"dev/loss", "dev/eer_percent", "eval/eer_percent"}
    if not required.issubset(history.columns):
        raise ValueError(f"Не найдены метрики: {sorted(required - set(history.columns))}")

    best_index = history.sort_values(
        ["dev/eer_percent", "dev/loss"],
        kind="stable",
    ).index[0]
    best_epoch = int(history.loc[best_index, "epoch"])
    best_dev = float(history.loc[best_index, "dev/eer_percent"])
    final_eval = float(history.loc[best_index, "eval/eer_percent"])

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        history["epoch"],
        history["dev/eer_percent"],
        marker="o",
        label="Dev EER",
    )
    axis.plot(
        history["epoch"],
        history["eval/eer_percent"],
        marker="o",
        label="Eval EER",
    )
    axis.scatter(
        [best_epoch],
        [best_dev],
        s=90,
        edgecolor="black",
        zorder=5,
        label=f"Лучший dev, эпоха {best_epoch}",
    )
    axis.scatter(
        [best_epoch],
        [final_eval],
        s=90,
        edgecolor="black",
        zorder=5,
        label=f"Eval выбранного checkpoint: {final_eval:.3f}%",
    )
    axis.axhline(5.3, linestyle="--", color="tab:purple", label="Граница 5.3%")
    axis.set(title="Dev и eval EER", xlabel="Эпоха", ylabel="EER, %")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    save_figure(figure, output_dir, "eer_by_epoch")

    print(
        f"Выбран checkpoint эпохи {best_epoch}: "
        f"dev EER={best_dev:.4f}%, eval EER={final_eval:.4f}%"
    )


def plot_optional_metrics(history: pd.DataFrame, output_dir: Path) -> None:
    for metric, title, ylabel, name in [
        ("learning_rate", "Learning rate", "Learning rate", "learning_rate"),
        (
            "dev/eer_threshold",
            "Dev EER threshold",
            "Порог score",
            "dev_eer_threshold",
        ),
    ]:
        if metric not in history:
            continue
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(history["epoch"], history[metric], marker="o")
        axis.set(title=title, xlabel="Эпоха", ylabel=ylabel)
        if metric == "learning_rate":
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
        save_figure(figure, output_dir, name)


def plot_per_attack(path: str, output_dir: Path) -> None:
    frame = pd.read_csv(path)
    required = {"attack_id", "eer_percent"}
    if not required.issubset(frame.columns):
        raise ValueError(f"В per-attack CSV нет колонок: {sorted(required - set(frame.columns))}")

    colors = [
        "tab:red" if attack_id in {"A17", "A18"} else "tab:green"
        for attack_id in frame["attack_id"]
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(frame["attack_id"], frame["eer_percent"], color=colors)
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axis.set(title="Eval EER по типам атак", xlabel="Тип атаки", ylabel="EER, %")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, output_dir, "per_attack_eer")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    history = read_history(args.history)
    history.to_csv(output_dir / "wandb_history_clean.csv", index=False)
    plot_losses(history, output_dir)
    plot_eer(history, output_dir)
    plot_optional_metrics(history, output_dir)
    if args.per_attack:
        plot_per_attack(args.per_attack, output_dir)
    print("Графики сохранены в", output_dir.resolve())


if __name__ == "__main__":
    main()
