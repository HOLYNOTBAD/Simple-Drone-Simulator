from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == SCRIPT_DIR:
	sys.path.pop(0)

try:
	import pandas as pd
except ImportError as e:
	raise SystemExit(
		"Failed to import pandas.\n"
		f"python executable: {sys.executable}\n"
		f"import error: {e}"
	) from e


REQUIRED_COLUMNS = ["VertexID", "X", "Y", "Z", "Time"]


def _load_source_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	return df


def _to_standard_csv(df: pd.DataFrame, id_prefix: str = "uav_") -> pd.DataFrame:
	out = df.loc[:, REQUIRED_COLUMNS].copy()
	out.rename(columns={"Time": "timestamp_ms"}, inplace=True)

	out["ID"] = out["VertexID"].apply(lambda x: f"{id_prefix}{int(float(x))}")
	out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
	out["X"] = pd.to_numeric(out["X"], errors="coerce")
	out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
	out["Z"] = pd.to_numeric(out["Z"], errors="coerce")

	out = out.dropna(subset=["timestamp_ms", "X", "Y", "Z", "ID"])
	out = out.loc[:, ["ID", "X", "Y", "Z", "timestamp_ms"]]
	out = out.sort_values(by=["ID", "timestamp_ms"], kind="stable").reset_index(drop=True)
	return out


def convert_csv(input_csv: Path, output_csv: Path, id_prefix: str = "uav_") -> pd.DataFrame:
	df = _load_source_csv(input_csv)
	standard_df = _to_standard_csv(df, id_prefix=id_prefix)
	standard_df.to_csv(output_csv, index=False)
	return standard_df


def _print_xyz_stats(df: pd.DataFrame) -> None:
	xyz_min = df[["X", "Y", "Z"]].min()
	xyz_max = df[["X", "Y", "Z"]].max()
	xyz_span = xyz_max - xyz_min

	print("XYZ 最小值:")
	print(f"  X_min: {xyz_min['X']}")
	print(f"  Y_min: {xyz_min['Y']}")
	print(f"  Z_min: {xyz_min['Z']}")
	print("XYZ 最大值:")
	print(f"  X_max: {xyz_max['X']}")
	print(f"  Y_max: {xyz_max['Y']}")
	print(f"  Z_max: {xyz_max['Z']}")
	print("XYZ 范围跨度:")
	print(f"  X_span: {xyz_span['X']}")
	print(f"  Y_span: {xyz_span['Y']}")
	print(f"  Z_span: {xyz_span['Z']}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert source trajectory CSV into standard multi-UAV control CSV format."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path(__file__).with_name("3.csv"),
		help="source CSV path, default: csv_process/3.csv",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).with_name("output.csv"),
		help="output CSV path, default: csv_process/output.csv",
	)
	parser.add_argument(
		"--id-prefix",
		type=str,
		default="uav_",
		help="prefix added before VertexID, default: uav_",
	)
	args = parser.parse_args()

	standard_df = convert_csv(args.input, args.output, id_prefix=args.id_prefix)

	print(f"转换完成，已保存到: {args.output}")
	print(f"总记录数: {len(standard_df)}")
	print(f"飞机数量: {standard_df['ID'].nunique()}")
	print("前 5 行预览:")
	print(standard_df.head().to_string(index=False))
	_print_xyz_stats(standard_df)


if __name__ == "__main__":
	main()