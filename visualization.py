import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the cleaned dataset
df = pd.read_csv("SWYNEX_cleaned_dataset.csv")

# Output folder for charts and report
Path("charts").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())

# Select numerical columns
numeric = df.select_dtypes(include="number")

if numeric.empty:
    raise ValueError("No numerical columns found in the dataset.")

# 1. Distribution plots
numeric.hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Distribution of Numerical Variables")
plt.tight_layout()
plt.savefig("charts/numerical_distributions.png", dpi=200)
plt.close()

# 2. Correlation heatmap
if numeric.shape[1] >= 2:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0
    )
    plt.title("Correlation Between Numerical Variables")
    plt.tight_layout()
    plt.savefig("charts/correlation_heatmap.png", dpi=200)
    plt.close()

# 3. Boxplots
for col in numeric.columns[:5]:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Distribution and Possible Outliers: {col}")
    plt.tight_layout()
    safe_name = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in col
    )
    plt.savefig(f"charts/boxplot_{safe_name}.png", dpi=200)
    plt.close()

# 4. Categorical frequency charts
categorical = df.select_dtypes(
    include=["object", "category", "bool"]
)

for col in categorical.columns[:3]:
    counts = df[col].value_counts().head(10)
    if not counts.empty:
        plt.figure(figsize=(9, 5))
        counts.plot(kind="bar", edgecolor="black")
        plt.title(f"Top Categories: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        safe_name = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in col
        )
        plt.savefig(f"charts/categories_{safe_name}.png", dpi=200)
        plt.close()

# 5. Save a scientific interpretation report
with open("INTERPRETATION.md", "w", encoding="utf-8") as f:
    f.write("# Task 3: Visualization and Interpretation\n\n")
    f.write("## Dataset overview\n\n")
    f.write(f"- Number of rows: {df.shape[0]}\n")
    f.write(f"- Number of columns: {df.shape[1]}\n")
    f.write(f"- Numerical variables: {numeric.shape[1]}\n\n")

    f.write("## Numerical summary\n\n")
    f.write(numeric.describe().round(3).to_markdown())
    f.write("\n\n## Interpretation of charts\n\n")
    f.write(
        "1. **Distribution plots:** Show the spread, concentration, "
        "and possible skewness of numerical values.\n"
    )
    f.write(
        "2. **Correlation heatmap:** Shows the direction and strength "
        "of linear associations between numerical variables. "
        "Correlation does not establish causation.\n"
    )
    f.write(
        "3. **Boxplots:** Show medians, quartiles, and potential "
        "outliers. A potential outlier is not automatically an error.\n"
    )
    f.write(
        "4. **Category charts:** Show the most frequent observed "
        "categories in the dataset.\n\n"
    )
    f.write("## Limitations\n\n")
    f.write(
        "- Visual patterns alone do not prove cause and effect.\n"
        "- Results describe this dataset and may not generalize "
        "to other populations.\n"
        "- Correlation may be affected by confounding variables "
        "or unusual observations.\n"
        "- Statistical significance and causal claims require "
        "additional appropriate analyses.\n"
    )

print("\nAnalysis completed successfully.")
print("Charts saved in the charts/ folder.")
print("Report saved as INTERPRETATION.md")
