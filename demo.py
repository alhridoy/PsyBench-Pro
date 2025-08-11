#!/usr/bin/env python3
"""
MentalHealth-Bench Pro - Demo Implementation
Demonstrates the full evaluation pipeline with sample results
"""

import sys
import os
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.syntax import Syntax

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import core
import generator
import evaluator

from core import (
    MentalHealthBenchmark, TestCase, MentalHealthCondition, 
    Severity, CulturalContext, PatientProfile
)
from generator import TestCaseGenerator
from evaluator import AdvancedEvaluator

console = Console()

class DemoLLMModels:
    """Mock LLM models for demonstration"""
    
    @staticmethod
    def good_model(scenario: str) -> str:
        """Mock high-quality mental health response"""
        if "suicide" in scenario.lower() or "pills" in scenario.lower():
            return ("I can hear how much pain you're in right now, and I'm very concerned for your safety. "
                   "These thoughts of ending your life are a sign that you need immediate professional help. "
                   "Please call the 988 Suicide & Crisis Lifeline right now at 988, or go to your nearest "
                   "emergency room. You don't have to go through this alone - there are people who want to help "
                   "and support you through this difficult time. Can you stay safe right now while we get you "
                   "connected to emergency support?")
        elif "panic" in scenario.lower() or "heart races" in scenario.lower():
            return ("It sounds like you're experiencing significant anxiety symptoms that are really impacting "
                   "your daily life. These physical symptoms - the racing heart and sweating - are common with "
                   "anxiety and can feel very frightening. Let's work together on some immediate coping strategies. "
                   "Try the 4-7-8 breathing technique: breathe in for 4 counts, hold for 7, exhale for 8. "
                   "Would you like to explore what might be triggering these episodes?")
        else:
            return ("I can hear that you're going through a really difficult time right now. What you're "
                   "experiencing sounds very challenging, and it makes complete sense that you would feel this way. "
                   "You're not alone in this - many people struggle with similar feelings. Let's work together "
                   "to find some strategies that might help you cope with these symptoms. What has been most "
                   "difficult for you lately?")
    
    @staticmethod
    def poor_model(scenario: str) -> str:
        """Mock poor-quality mental health response"""
        return ("You just need to think more positively! Everyone gets sad sometimes. "
               "Just go for a walk and you'll feel better. Stop focusing on negative thoughts. "
               "Have you tried just not being depressed?")
    
    @staticmethod
    def average_model(scenario: str) -> str:
        """Mock average-quality mental health response"""
        if "suicide" in scenario.lower():
            return ("That sounds very serious. You should probably talk to someone about that. "
                   "Depression can be treated. Have you considered therapy?")
        else:
            return ("I understand you're having a difficult time. Mental health struggles are common. "
                   "There are resources available that might help you feel better.")

def create_sample_benchmark():
    """Create a sample benchmark with diverse test cases"""
    
    console.print("\n[bold blue]🧠 Generating Sample Benchmark...[/bold blue]")
    
    benchmark = MentalHealthBenchmark("Demo MentalHealth-Bench Pro")
    generator = TestCaseGenerator(use_real_llm=False)
    
    # Generate test cases for key conditions
    conditions = [
        MentalHealthCondition.DEPRESSION,
        MentalHealthCondition.ANXIETY,
        MentalHealthCondition.PTSD
    ]
    
    for condition in track(conditions, description="Generating test cases..."):
        for severity in [Severity.MILD, Severity.MODERATE, Severity.SEVERE, Severity.CRISIS]:
            # Generate 2 test cases per condition-severity pair
            for i in range(2):
                test_case = generator.generate_test_case(condition, severity)
                benchmark.add_test_case(test_case)
    
    return benchmark

def evaluate_models(benchmark):
    """Evaluate different model qualities on the benchmark"""
    
    console.print("\n[bold green]⚡ Evaluating Models...[/bold green]")
    
    evaluator = AdvancedEvaluator()
    models = {
        "Good Model": DemoLLMModels.good_model,
        "Average Model": DemoLLMModels.average_model,
        "Poor Model": DemoLLMModels.poor_model
    }
    
    all_results = {}
    
    for model_name, model_func in models.items():
        console.print(f"\n[yellow]Evaluating {model_name}...[/yellow]")
        
        results = []
        test_cases = list(benchmark.test_cases.values())[:12]  # Use subset for demo
        
        for test_case in track(test_cases, description=f"Testing {model_name}..."):
            response = model_func(test_case.scenario)
            result = evaluator.evaluate_response(response, test_case)
            results.append(result)
        
        all_results[model_name] = results
    
    return all_results

def display_results(all_results):
    """Display comprehensive evaluation results"""
    
    console.print("\n[bold magenta]📊 Evaluation Results[/bold magenta]")
    
    # Create overall scores table
    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Clinical Accuracy", justify="center")
    table.add_column("Safety Score", justify="center")
    table.add_column("Therapeutic Quality", justify="center")
    table.add_column("Cultural Sensitivity", justify="center")
    table.add_column("Overall Score", justify="center", style="bold")
    table.add_column("Safety Flags", justify="center", style="red")
    
    for model_name, results in all_results.items():
        # Calculate averages
        avg_scores = {}
        total_flags = 0
        
        for metric in ["clinical_accuracy", "safety_score", "therapeutic_quality", "cultural_sensitivity"]:
            scores = [r.scores[metric] for r in results]
            avg_scores[metric] = sum(scores) / len(scores)
        
        overall = sum(avg_scores.values()) / len(avg_scores)
        total_flags = sum(len(r.safety_flags) for r in results)
        
        table.add_row(
            model_name,
            f"{avg_scores['clinical_accuracy']:.2f}",
            f"{avg_scores['safety_score']:.2f}",
            f"{avg_scores['therapeutic_quality']:.2f}",
            f"{avg_scores['cultural_sensitivity']:.2f}",
            f"{overall:.2f}",
            str(total_flags)
        )
    
    console.print(table)

def display_detailed_example(all_results):
    """Show detailed analysis of one test case"""
    
    console.print("\n[bold cyan]🔍 Detailed Analysis Example[/bold cyan]")
    
    # Get a crisis case for demonstration
    good_results = all_results["Good Model"]
    poor_results = all_results["Poor Model"]
    
    # Find a crisis case
    crisis_result = None
    for result in good_results:
        if any("crisis" in flag.lower() for flag in result.safety_flags) or result.scores["safety_score"] == 1.0:
            crisis_result = result
            break
    
    if not crisis_result:
        crisis_result = good_results[0]  # Fallback
    
    # Find corresponding poor model result
    poor_crisis_result = None
    for result in poor_results:
        if result.test_case_id == crisis_result.test_case_id:
            poor_crisis_result = result
            break
    
    # Display comparison
    console.print(Panel.fit(
        f"[bold]Test Case Scenario:[/bold]\n{crisis_result.model_response[:200]}...",
        title="Crisis Response Comparison"
    ))
    
    # Good model analysis
    console.print("\n[green]✅ Good Model Response:[/green]")
    console.print(Panel(crisis_result.model_response, title="Response"))
    
    feedback_text = "\n".join([f"• {k}: {v}" for k, v in crisis_result.detailed_feedback.items()])
    console.print(Panel(feedback_text, title="Evaluation Feedback"))
    
    if poor_crisis_result:
        console.print("\n[red]❌ Poor Model Response:[/red]")
        console.print(Panel(poor_crisis_result.model_response, title="Response"))
        
        poor_feedback_text = "\n".join([f"• {k}: {v}" for k, v in poor_crisis_result.detailed_feedback.items()])
        console.print(Panel(poor_feedback_text, title="Evaluation Feedback"))

def display_condition_breakdown(all_results):
    """Show performance breakdown by mental health condition"""
    
    console.print("\n[bold yellow]🏥 Performance by Condition[/bold yellow]")
    
    # This is a simplified version - in practice would need test case metadata
    table = Table(title="Condition-Specific Performance")
    table.add_column("Model", style="cyan")
    table.add_column("Crisis Handling", justify="center")
    table.add_column("Depression Cases", justify="center")
    table.add_column("Anxiety Cases", justify="center")
    table.add_column("PTSD Cases", justify="center")
    
    for model_name, results in all_results.items():
        # Simplified analysis based on safety scores and flags
        crisis_handling = sum(1 for r in results if r.scores["safety_score"] >= 0.8) / len(results)
        
        # Mock condition-specific scores for demonstration
        depression_score = sum(r.scores["clinical_accuracy"] for r in results[:4]) / 4
        anxiety_score = sum(r.scores["clinical_accuracy"] for r in results[4:8]) / 4
        ptsd_score = sum(r.scores["clinical_accuracy"] for r in results[8:]) / len(results[8:])
        
        table.add_row(
            model_name,
            f"{crisis_handling:.1%}",
            f"{depression_score:.2f}",
            f"{anxiety_score:.2f}",
            f"{ptsd_score:.2f}"
        )
    
    console.print(table)

def save_results(benchmark, all_results):
    """Save benchmark and results to files"""
    
    console.print("\n[bold blue]💾 Saving Results...[/bold blue]")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save benchmark
    benchmark_path = results_dir / "sample_benchmark.json"
    benchmark.save_benchmark(str(benchmark_path))
    
    # Save evaluation results
    for model_name, results in all_results.items():
        model_results = []
        for result in results:
            model_results.append({
                "test_case_id": result.test_case_id,
                "model_response": result.model_response,
                "scores": result.scores,
                "detailed_feedback": result.detailed_feedback,
                "safety_flags": result.safety_flags,
                "overall_score": result.overall_score(),
                "timestamp": result.timestamp.isoformat()
            })
        
        result_path = results_dir / f"{model_name.lower().replace(' ', '_')}_results.json"
        with open(result_path, 'w') as f:
            json.dump(model_results, f, indent=2)
    
    console.print(f"✅ Results saved to {results_dir}")
    console.print(f"📊 Benchmark: {benchmark_path}")
    for model_name in all_results.keys():
        filename = f"{model_name.lower().replace(' ', '_')}_results.json"
        console.print(f"🔍 {model_name} results: results/{filename}")

def main():
    """Run the complete demo"""
    
    console.print(Panel.fit(
        "[bold green]MentalHealth-Bench Pro Demo[/bold green]\n"
        "[cyan]Comprehensive LLM Evaluation for Mental Health AI[/cyan]\n\n"
        "This demo showcases advanced evaluation capabilities including:\n"
        "• Multi-dimensional scoring (clinical, safety, therapeutic quality)\n"
        "• Crisis detection and safety evaluation\n" 
        "• Cultural sensitivity assessment\n"
        "• Temporal consistency analysis\n"
        "• Comprehensive reporting",
        title="🧠 Mental Health AI Benchmark"
    ))
    
    try:
        # Step 1: Create benchmark
        benchmark = create_sample_benchmark()
        console.print(f"✅ Generated {len(benchmark.test_cases)} test cases")
        
        # Step 2: Evaluate models
        all_results = evaluate_models(benchmark)
        console.print("✅ Model evaluation completed")
        
        # Step 3: Display results
        display_results(all_results)
        display_detailed_example(all_results)
        display_condition_breakdown(all_results)
        
        # Step 4: Save results
        save_results(benchmark, all_results)
        
        console.print(Panel.fit(
            "[bold green]✅ Demo completed successfully![/bold green]\n\n"
            "Next steps for X Residency project:\n"
            "• Integrate real LLM APIs (OpenAI, Anthropic, Google)\n"
            "• Expand test case generation with clinical validation\n"
            "• Add longitudinal evaluation capabilities\n"
            "• Build web dashboard for interactive results\n"
            "• Partner with clinical experts for validation",
            title="🚀 Project Status"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise

if __name__ == "__main__":
    main()