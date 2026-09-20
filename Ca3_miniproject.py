"""
Project Title: Autonomous Multi-Tool Reasoning Agent for Demand Forecasting
Course: Agentic AI & Automation (AY 2026-27)
Institution: Symbiosis Institute of Technology, Nagpur
Author: Madhura Dnyaneshwar Rewatkar (PRN: 24070521037)

Description:
A complete ReAct-pattern autonomous AI agent that ingests time-series sales,
executes deterministic trend-forecasting and inventory replenishment algorithms,
and generates structured procurement decision reports without arithmetic hallucination.
"""

import ast
import json
import operator
import os
import sys
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI


# =====================================================================
# 1. TIME-SERIES REPOSITORY & DETERMINISTIC SUPPLY CHAIN TOOLS
# =====================================================================

# Simulated ERP telemetry: weekly historical unit sales
HISTORICAL_SALES_DB: Dict[str, List[float]] = {
    "SKU-101": [120.0, 135.0, 128.0, 142.0, 150.0, 165.0, 170.0, 180.0],  # Rapid acceleration (+8.6 units/wk)
    "SKU-202": [300.0, 290.0, 310.0, 295.0, 305.0, 300.0, 298.0, 302.0],  # Stationary / Constant (~300 units/wk)
    "SKU-303": [80.0, 75.0, 70.0, 65.0, 60.0, 50.0, 45.0, 40.0],          # Sharp decline (-5.7 units/wk)
}


class SecureASTMathEvaluator:
    """Safely computes arithmetic strings without using unsafe eval()."""

    _valid_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    @classmethod
    def evaluate(cls, expression: str) -> float:
        try:
            tree = ast.parse(expression.strip(), mode="eval").body
            return cls._traverse(tree)
        except Exception as exc:
            raise ValueError(f"Invalid arithmetic expression '{expression}': {str(exc)}")

    @classmethod
    def _traverse(cls, node: Any) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in cls._valid_operators:
            left_val = cls._traverse(node.left)
            right_val = cls._traverse(node.right)
            return cls._valid_operators[type(node.op)](left_val, right_val)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in cls._valid_operators:
            return cls._valid_operators[type(node.op)](cls._traverse(node.operand))
        else:
            raise TypeError(f"Disallowed syntax node: {type(node).__name__}")


def calculate_demand_forecast(sku_id: str, forecast_periods: int = 4) -> str:
    """Tool: Ingests historical sales, calculates moving average, slope, and projects demand."""
    clean_sku = sku_id.strip().upper()
    sales = HISTORICAL_SALES_DB.get(clean_sku)

    if not sales:
        return json.dumps({
            "status": "error",
            "message": f"SKU '{clean_sku}' not found in ERP repository. Available SKUs: {list(HISTORICAL_SALES_DB.keys())}"
        })

    n = len(sales)
    # Moving average across the latest 3 periods
    recent_window = min(3, n)
    moving_avg_recent = sum(sales[-recent_window:]) / recent_window

    # Linear trend slope: (S_n - S_1) / (n - 1)
    trend_slope = (sales[-1] - sales[0]) / (n - 1)

    # Dynamic projection for the next N periods
    projections = [
        round(max(0.0, sales[-1] + (trend_slope * i)), 2)
        for i in range(1, forecast_periods + 1)
    ]
    total_projected_demand = round(sum(projections), 2)

    output = {
        "sku": clean_sku,
        "historical_periods_evaluated": n,
        "recent_sales_velocity": sales[-3:],
        "recent_3_period_moving_avg": round(moving_avg_recent, 2),
        "weekly_trend_slope": round(trend_slope, 2),
        "trend_trajectory": "ACCELERATING" if trend_slope > 1.0 else ("DECLINING" if trend_slope < -1.0 else "STEADY"),
        "forecast_periods": forecast_periods,
        "period_projections": projections,
        "total_forecasted_demand": total_projected_demand,
        "average_weekly_demand": round(total_projected_demand / forecast_periods, 2),
    }
    return json.dumps(output)


def calculate_inventory_reorder(
    projected_demand: float,
    lead_time_weeks: int = 2,
    current_stock: int = 150,
    buffer_percentage: float = 0.50
) -> str:
    """Tool: Evaluates warehouse stock, computes safety stock, reorder point (ROP), and quantity."""
    # Assuming standard 4-week planning window
    weekly_burn_rate = projected_demand / 4.0

    # Safety Stock (SS) = Weekly Burn * Buffer Percentage
    safety_stock = round(weekly_burn_rate * buffer_percentage, 2)

    # Reorder Point (ROP) = (Weekly Demand * Lead Time) + Safety Stock
    reorder_point = round((weekly_burn_rate * lead_time_weeks) + safety_stock, 2)

    # Decision logic
    needs_replenishment = current_stock <= reorder_point
    recommended_order_quantity = 0.0

    if needs_replenishment:
        # Target Stock Level = Total Period Demand + Safety Stock
        target_inventory = projected_demand + safety_stock
        recommended_order_quantity = max(0.0, round(target_inventory - current_stock, 2))

    output = {
        "current_warehouse_stock": current_stock,
        "lead_time_weeks": lead_time_weeks,
        "weekly_burn_rate": round(weekly_burn_rate, 2),
        "computed_safety_stock": safety_stock,
        "reorder_point_threshold": reorder_point,
        "replenishment_trigger": "REORDER_NOW" if needs_replenishment else "STOCK_SUFFICIENT",
        "recommended_order_quantity": int(recommended_order_quantity),
        "stockout_risk_profile": "HIGH" if current_stock < (weekly_burn_rate * lead_time_weeks) else "LOW_TO_MODERATE",
    }
    return json.dumps(output)


def compute_safe_math(expression: str) -> str:
    """Tool: Executes arithmetic computations deterministically via AST."""
    try:
        res = SecureASTMathEvaluator.evaluate(expression)
        return json.dumps({"status": "success", "expression": expression, "result": res})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# =====================================================================
# 2. OPENAI FUNCTION SCHEMAS & TOOL REGISTRY
# =====================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_demand_forecast",
            "description": "Calculates statistical time-series moving averages, trend slopes, and projected sales demand for a SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku_id": {
                        "type": "string",
                        "description": "The SKU code to look up (e.g., 'SKU-101', 'SKU-202', 'SKU-303')."
                    },
                    "forecast_periods": {
                        "type": "integer",
                        "description": "Number of future weeks to project (default is 4)."
                    }
                },
                "required": ["sku_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_inventory_reorder",
            "description": "Determines safety stock, reorder point thresholds, and optimal replenishment order sizes based on projected demand.",
            "parameters": {
                "type": "object",
                "properties": {
                    "projected_demand": {
                        "type": "number",
                        "description": "The total projected demand units over the forecast window."
                    },
                    "lead_time_weeks": {
                        "type": "integer",
                        "description": "Supplier fulfillment lead time in weeks."
                    },
                    "current_stock": {
                        "type": "integer",
                        "description": "Current on-hand units available in the warehouse."
                    },
                    "buffer_percentage": {
                        "type": "number",
                        "description": "Safety margin multiplier (default is 0.50 for a 50% buffer)."
                    }
                },
                "required": ["projected_demand", "lead_time_weeks", "current_stock"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_safe_math",
            "description": "Executes arbitrary mathematical expressions (+, -, *, /, **) safely without hallucination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression string, e.g., '(771.4 + 96.4) - 150'"
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "calculate_demand_forecast": calculate_demand_forecast,
    "calculate_inventory_reorder": calculate_inventory_reorder,
    "compute_safe_math": compute_safe_math,
}


# =====================================================================
# 3. AUTONOMOUS REACT AGENT ORCHESTRATOR
# =====================================================================

class DemandForecastingAgent:
    """Autonomous Supply Chain Agent using the ReAct (Reason + Act) loop."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        self.system_prompt = (
            "You are an autonomous Supply Chain & Demand Forecasting AI Agent.\n"
            "Your role:\n"
            "1. Analyze inventory and demand tasks by planning sequential steps.\n"
            "2. Always call 'calculate_demand_forecast' first to evaluate sales trends and project future demand.\n"
            "3. Always call 'calculate_inventory_reorder' to calculate safety stocks and replenishment orders.\n"
            "4. NEVER perform mathematical computations yourself. Use the provided tools.\n"
            "5. Synthesize a clean, professional replenishment memo clearly stating the metrics, trigger status, and final order recommendation."
        )

    def _execute_local_simulation(self, user_query: str) -> str:
        """Deterministic offline fallback for review demos without API keys."""
        print("\n[NOTE]: OPENAI_API_KEY not detected. Executing deterministic local ReAct simulation...\n")

        # Step 1: Mock tool execution for SKU-101
        print("-> [Agent Plan]: Querying time-series trends for SKU-101...")
        forecast_raw = calculate_demand_forecast("SKU-101", forecast_periods=4)
        forecast_data = json.loads(forecast_raw)
        print(f"<- [Observation - Forecast]: {forecast_raw}\n")

        # Step 2: Mock tool execution for inventory policy
        print("-> [Agent Plan]: Evaluating inventory thresholds (Stock: 150, Lead Time: 2 weeks)...")
        inventory_raw = calculate_inventory_reorder(
            projected_demand=forecast_data["total_forecasted_demand"],
            lead_time_weeks=2,
            current_stock=150,
        )
        inv_data = json.loads(inventory_raw)
        print(f"<- [Observation - Inventory]: {inventory_raw}\n")

        # Synthesized final report
        return (
            "=====================================================================\n"
            "EXECUTIVE DEMAND & REPLENISHMENT REPORT (SKU-101)\n"
            "=====================================================================\n"
            f"1. HISTORICAL DEMAND TRAJECTORY:\n"
            f"   - Recent 3-Week Moving Average : {forecast_data['recent_3_period_moving_avg']} units\n"
            f"   - Weekly Trend Slope           : +{forecast_data['weekly_trend_slope']} units/week ({forecast_data['trend_trajectory']})\n"
            f"   - 4-Week Period Forecast       : {forecast_data['period_projections']}\n"
            f"   - Total Projected Demand       : {forecast_data['total_forecasted_demand']} units\n\n"
            f"2. INVENTORY HEALTH & REORDER ASSESSMENT:\n"
            f"   - Current Warehouse Stock      : {inv_data['current_warehouse_stock']} units\n"
            f"   - Supplier Lead Time           : {inv_data['lead_time_weeks']} weeks\n"
            f"   - Reorder Point Threshold (ROP): {inv_data['reorder_point_threshold']} units\n"
            f"   - Safety Stock Allocated       : {inv_data['computed_safety_stock']} units\n"
            f"   - Stockout Risk Profile        : {inv_data['stockout_risk_profile']}\n\n"
            f"3. DECISION RECOMMENDATION:\n"
            f"   - Action Trigger               : [{inv_data['replenishment_trigger']}]\n"
            f"   - Recommended Order Quantity   : {inv_data['recommended_order_quantity']} units\n"
            f"   - Operational Justification    : Current stock (150 units) is significantly lower than\n"
            f"     the reorder threshold ({inv_data['reorder_point_threshold']} units). Immediate order\n"
            f"     placement is required to sustain customer fulfillment during the 2-week lead time.\n"
            "====================================================================="
        )

    def run(self, user_goal: str, max_iterations: int = 6) -> str:
        """Main agent loop executing the Sense-Plan-Act cycle."""
        if not self.client:
            return self._execute_local_simulation(user_goal)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_goal},
        ]

        print(f"\n[AGENT SESSION INITIALIZED]: {user_goal}")
        print("=" * 70)

        for iteration in range(1, max_iterations + 1):
            print(f"\n[Iteration {iteration}] Agent Reasoning...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if the agent wants to call external tools
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    print(f"  -> Action: Call '{function_name}'")
                    print(f"     Payload: {raw_args}")

                    try:
                        kwargs = json.loads(raw_args)
                    except json.JSONDecodeError:
                        kwargs = {}

                    executor = TOOL_REGISTRY.get(function_name)
                    if executor:
                        try:
                            observation = executor(**kwargs)
                        except Exception as err:
                            observation = json.dumps({"status": "execution_error", "details": str(err)})
                    else:
                        observation = json.dumps({"status": "error", "message": f"Tool '{function_name}' not found."})

                    print(f"  <- Observation: {observation}")

                    # Return observation back to conversation memory
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(observation),
                    })
            else:
                # Terminal condition: agent arrived at final synthesized conclusion
                print("\n[GOAL REACHED]: Autonomous plan completed.")
                print("=" * 70)
                return assistant_message.content or ""

        return "[HALTED]: Reached maximum allowed ReAct iterations without convergence."


# =====================================================================
# 4. ENTRYPOINT & TESTING SUITE
# =====================================================================

if __name__ == "__main__":
    agent = DemandForecastingAgent()

    test_prompt = (
        "Perform an operational demand forecast for SKU-101 for the next 4 weeks. "
        "Our warehouse currently holds 150 units, and the supplier takes 2 weeks to fulfill orders. "
        "Determine if we need to place a reorder immediately, compute our safety stock requirements, "
        "and state the exact number of units we should purchase."
    )

    final_output = agent.run(test_prompt)
    print("\n" + final_output)
      
