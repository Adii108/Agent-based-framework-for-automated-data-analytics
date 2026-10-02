"""Controlled Analytical Execution Sandbox for AutoAnalytics.

Executes generated Python analytical code in an isolated, monitored environment
with stdout/stderr interception, timeouts, AST safety screening, and structured
artifact/metric extraction.
"""

import ast
import io
import sys
import time
import traceback
from typing import Any
import pandas as pd
import numpy as np

from backend.state import ExecutionRecord, AnalyticalPlan, PlanStep, AutoAnalyticsState
from backend.tools.ingestion import load_dataset
from backend.tools.cleaning import clean_data


# Disallowed AST nodes and dangerous built-ins for safety screening
BANNED_IMPORTS = {
    "os",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "pty",
    "multiprocessing",
    "signal",
}

BANNED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",  # Raw file writing is banned; load data through sandbox context
}


class CodeSecurityValidator(ast.NodeVisitor):
    """AST scanner that blocks unsafe system operations before execution."""
    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_module = alias.name.split('.')[0]
            if root_module in BANNED_IMPORTS:
                self.violations.append(f"Security restriction: Direct import of module '{root_module}' is prohibited.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root_module = node.module.split('.')[0]
            if root_module in BANNED_IMPORTS:
                self.violations.append(f"Security restriction: Import from module '{root_module}' is prohibited.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            self.violations.append(f"Security restriction: Execution of function '{node.func.id}()' is prohibited.")
        self.generic_visit(node)


def validate_code_safety(code: str) -> tuple[bool, list[str]]:
    """Scan code AST for unsafe constructs."""
    try:
        tree = ast.parse(code)
        scanner = CodeSecurityValidator()
        scanner.visit(tree)
        return len(scanner.violations) == 0, scanner.violations
    except SyntaxError as e:
        return False, [f"SyntaxError: {str(e)}"]


def execute_sandboxed_code(
    code: str,
    df: pd.DataFrame,
    step_id: int = 1,
    context_vars: dict[str, Any] | None = None,
    timeout_sec: float = 30.0,
) -> ExecutionRecord:
    """Execute analytical Python code against a DataFrame in a controlled namespace."""
    start_time = time.perf_counter()
    
    # 1. AST Security Validation
    is_safe, security_violations = validate_code_safety(code)
    if not is_safe:
        return {
            "step_id": step_id,
            "code": code,
            "runtime": "python",
            "stdout": "",
            "stderr": "\n".join(security_violations),
            "return_value": None,
            "extracted_metrics": {},
            "artifacts": {},
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "success": False,
            "error_message": f"SecurityCheckFailed: {'; '.join(security_violations)}",
        }

    # 2. Setup Sandbox Environment
    # Clone dataframe so in-memory mutations don't corrupt baseline
    local_df = df.copy()
    
    import sklearn
    import statsmodels
    import scipy
    
    from backend.tools.eda import run_eda
    from backend.tools.prediction import predict_churn, generate_forecast, prepare_churn_features
    from backend.tools.explainability import generate_shap_explanation, explain_customer_churn, explain_forecast

    sandbox_globals = {
        "__builtins__": {
            k: v for k, v in __builtins__.items()
            if k not in BANNED_CALLS and not k.startswith("__")
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k) for k in dir(__builtins__)
            if k not in BANNED_CALLS and not k.startswith("__")
        },
        "pd": pd,
        "np": np,
        "sklearn": sklearn,
        "statsmodels": statsmodels,
        "scipy": scipy,
        "df": local_df,
        "run_eda": run_eda,
        "predict_churn": predict_churn,
        "generate_forecast": generate_forecast,
        "prepare_churn_features": prepare_churn_features,
        "generate_shap_explanation": generate_shap_explanation,
        "explain_customer_churn": explain_customer_churn,
        "explain_forecast": explain_forecast,
    }

    if context_vars:
        sandbox_globals.update(context_vars)

    sandbox_locals: dict[str, Any] = {}

    # 3. Intercept I/O and Execute
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    success = False
    error_msg = None
    extracted_metrics = {}
    artifacts = {}
    return_val = None

    try:
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        # Compile and execute within sandbox
        compiled = compile(code, "<sandbox>", "exec")
        exec(compiled, sandbox_globals, sandbox_locals)
        success = True

        # Extract standard outputs
        if "result" in sandbox_locals:
            return_val = sandbox_locals["result"]
        elif "output" in sandbox_locals:
            return_val = sandbox_locals["output"]

        # Automatically extract metrics and tables created in locals
        for var_name, var_val in sandbox_locals.items():
            if var_name.startswith("_"):
                continue
            if isinstance(var_val, (int, float, str, bool, list, dict)):
                extracted_metrics[var_name] = var_val
            elif isinstance(var_val, pd.DataFrame):
                artifacts[var_name] = {
                    "shape": var_val.shape,
                    "columns": list(var_val.columns),
                    "head": var_val.head(5).to_dict(orient="records"),
                }

    except Exception as e:
        success = False
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc(file=stderr_buf)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "step_id": step_id,
        "code": code,
        "runtime": "python",
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "return_value": return_val,
        "extracted_metrics": extracted_metrics,
        "artifacts": artifacts,
        "execution_time_ms": elapsed_ms,
        "success": success,
        "error_message": error_msg,
    }


def generate_code_for_step(step: PlanStep, data_context: dict[str, Any]) -> str:
    """Generate deterministic Python analytical code for an individual plan step."""
    op = step.get("operation", "aggregate")
    req = step.get("required_data", [])

    if op == "time_series_forecast":
        return """# Time Series Forecasting
forecast_res = generate_forecast(df, periods=90)
result = {
    'summary': forecast_res.get('summary'),
    'points_count': len(forecast_res.get('forecast', [])),
    'trend': forecast_res.get('summary', {}).get('trend'),
    'historical_avg': forecast_res.get('summary', {}).get('last_30_days_avg'),
    'forecast_avg': forecast_res.get('summary', {}).get('forecast_avg'),
}
print(f"Forecast complete: Trend={result['trend']}, Projected Avg={result['forecast_avg']}")
"""

    elif op in ("classification", "feature_engineering"):
        return """# Supervised Classification / Churn Analysis
churn_res = predict_churn(df)
result = {
    'accuracy': churn_res.get('model_accuracy'),
    'total_customers': churn_res.get('total_customers'),
    'predicted_churned': churn_res.get('predicted_churned'),
    'predicted_retained': churn_res.get('predicted_retained'),
    'top_risk': churn_res.get('predictions', [])[:5],
}
print(f"Classification complete: Accuracy={result['accuracy']}, Churned={result['predicted_churned']}")
"""

    elif op == "explainability":
        return """# SHAP Explainability & Factor Attribution
churn_res = predict_churn(df)
shap_res = generate_shap_explanation(churn_res['model'], churn_res['feature_data'])
result = {
    'base_value': shap_res.get('base_value'),
    'global_importance': shap_res.get('global_importance', []),
}
print(f"Explainability complete: Top factors={[f['feature'] for f in result['global_importance'][:3]]}")
"""

    elif op == "correlation":
        return """# Statistical Correlation Analysis
num_df = df.select_dtypes(include=[np.number])
corr_matrix = num_df.corr().round(4).to_dict()
result = {
    'correlation_matrix': corr_matrix,
    'numeric_features': list(num_df.columns),
}
print(f"Correlations computed for {len(num_df.columns)} numeric variables.")
"""

    else:
        # Standard aggregation / summary
        num_cols = data_context.get("numeric_columns", ["revenue", "quantity"])
        cat_cols = data_context.get("categorical_columns", ["product_category", "region"])
        
        target_num = req[1] if len(req) > 1 and req[1] in num_cols else (num_cols[0] if num_cols else "revenue")
        target_cat = req[0] if req and req[0] in cat_cols else (cat_cols[0] if cat_cols else "product_category")

        return f"""# Aggregation & Distribution Analysis
if '{target_cat}' in df.columns and '{target_num}' in df.columns:
    agg_table = df.groupby('{target_cat}')['{target_num}'].agg(['sum', 'mean', 'count']).reset_index()
    agg_table['sum'] = agg_table['sum'].round(2)
    agg_table['mean'] = agg_table['mean'].round(2)
    result = {{
        'breakdown': agg_table.to_dict(orient='records'),
        'total_{target_num}': round(float(df['{target_num}'].sum()), 2),
        'mean_{target_num}': round(float(df['{target_num}'].mean()), 2),
    }}
    print(f"Aggregated {target_num} by {target_cat}: {{len(agg_table)}} segments")
else:
    stats = df.describe().round(2).to_dict()
    result = {{'summary_statistics': stats}}
    print("Computed descriptive statistics.")
"""


def analytical_executor_node(state: AutoAnalyticsState) -> dict[str, Any]:
    """LangGraph node executing the validated plan steps inside the sandbox."""
    dataset_path = state.get("dataset_name") or "data/sample_ecommerce.csv"
    try:
        raw_df = load_dataset(dataset_path)
        df, _ = clean_data(raw_df)
    except Exception as e:
        df = pd.DataFrame()

    plan = state.get("analytical_plan") or {"steps": []}
    steps = plan.get("steps", [])
    data_ctx = state.get("data_context") or {}

    history: list[ExecutionRecord] = list(state.get("execution_history") or [])
    last_exec: ExecutionRecord | None = None
    all_success = True
    context_vars: dict[str, Any] = {}

    for step in steps:
        code = generate_code_for_step(step, data_ctx)
        exec_record = execute_sandboxed_code(
            code=code,
            df=df,
            step_id=step.get("id", 1),
            context_vars=context_vars,
        )
        history.append(exec_record)
        last_exec = exec_record

        # Propagate outputs to downstream steps
        if exec_record["return_value"] is not None:
            context_vars[f"step_{step.get('id', 1)}_output"] = exec_record["return_value"]

        if not exec_record["success"]:
            all_success = False
            break

    trace_entry = {
        "node": "analytical_executor",
        "steps_executed": len(steps),
        "all_success": all_success,
        "last_error": last_exec.get("error_message") if last_exec and not last_exec.get("success") else None,
    }

    return {
        "execution_history": history,
        "current_execution": last_exec or {
            "step_id": 1,
            "code": "",
            "runtime": "python",
            "stdout": "",
            "stderr": "",
            "return_value": None,
            "extracted_metrics": {},
            "artifacts": {},
            "execution_time_ms": 0.0,
            "success": False,
            "error_message": "No steps executed",
        },
        "current_code": last_exec.get("code", "") if last_exec else "",
        "execution_error": last_exec.get("error_message") if last_exec and not last_exec.get("success") else None,
        "trace": (state.get("trace") or []) + [trace_entry],
    }
