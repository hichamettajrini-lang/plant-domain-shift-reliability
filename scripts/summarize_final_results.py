import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = root / "results"

final = pd.read_csv(results / "final_domain_adaptation_results_resnet50.csv")
q1 = pd.read_csv(results / "q1_metrics_augmented_finetuning_resnet50.csv")
bootstrap = pd.read_csv(results / "bootstrap_ci_resnet50_plantdoc_adaptation_fixed_38_labels.csv")

print("Final domain adaptation results")
print(final)
print("
Best adapted model metrics")
print(q1)
print("
Bootstrap confidence intervals")
print(bootstrap)
