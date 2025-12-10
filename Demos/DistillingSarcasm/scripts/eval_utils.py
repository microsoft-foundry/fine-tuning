# Original version from:
# https://raw.githubusercontent.com/azure-ai-foundry/build-2025-demos/refs/heads/main/Azure%20AI%20Model%20Customization/DistillationDemo/scripts/eval_utils.py
import openai
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for percentile calculations

from concurrent.futures import ThreadPoolExecutor, as_completed

def get_eval_runs_list(client: openai.Client, eval_id: str) -> list:
    """
    Fetch the list of evaluation runs for a given evaluation ID.

    Args:
        eval_id (str): The evaluation ID.

    Returns:
        list: A list of evaluation runs with their details.
    """
    runs = client.evals.runs.list(eval_id)

    print(f"Get Evaluation Runs: {eval_id}")
    list_runs = []

    if runs:
        for run in runs:
            r = {
                'id': run.id,
                'name': run.name,
                'status': run.status,
                'model': run.model,
                'pass_percentage': 0,
                'error_percentage': 0,
            }
            result = run.result_counts
            if result:
                pass_percentage = (result.passed * 100) / (result.passed + result.failed) if (result.passed + result.failed) > 0 else 0
                error_percentage = (result.errored * 100) / result.total if result.total > 0 else 0
                r['pass_percentage'] = pass_percentage
                r['error_percentage'] = error_percentage

            list_runs.append(r)

    return list_runs


def get_eval_details(client: openai.Client, eval_id: str) -> dict:
    """
    Fetch the details of a specific evaluation.

    Args:
        eval_id (str): The evaluation ID.

    Returns:
        dict: A dictionary containing evaluation details, including the name.
    """
    try:
        eval = client.evals.retrieve(eval_id)
        return eval.to_dict()
    except Exception as e:
        print(f"Failed to fetch evaluation details for ID: {eval_id}. Error: {e}")
        return {"name": f"Unknown Evaluation ({eval_id})"}


def display_evaluation_summary(client: openai.Client, eval_ids: list, x_range = (0, 7)):
    """
    Fetch and display a summary of evaluation runs for a list of evaluation IDs, including a horizontal bar chart,
    average score, and score distribution for all runs in a single chart with a maximum of 4 graphs per row.

    Args:
        eval_ids (list): A list of evaluation IDs.
    """
    all_eval_runs = []
    eval_id_to_name = {}
    eval_id_to_color = {}

    # Assign unique colors for each evaluation ID
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    for i, eval_id in enumerate(eval_ids):
        eval_id_to_color[eval_id] = colors[i % len(colors)]

    # Fetch evaluation runs and details for each evaluation ID
    for eval_id in eval_ids:
        eval_runs = get_eval_runs_list(client, eval_id)

        # Fetch evaluation details using the helper method
        eval_details = get_eval_details(client, eval_id)
        eval_name = eval_details.get('name', f'Unknown Evaluation ({eval_id})')
        eval_id_to_name[eval_id] = eval_name

        # Add evaluation ID to each run for color coding
        for run in eval_runs:
            run['eval_id'] = eval_id
            all_eval_runs.append(run)

    # Combine all evaluation runs into a single DataFrame
    if all_eval_runs:
        df = pd.DataFrame(all_eval_runs)
        df = df[['id', 'name', 'model', 'status', 'pass_percentage', 'error_percentage', 'eval_id']]  # Select relevant columns
        df['eval_name'] = df['eval_id'].map(eval_id_to_name)  # Map eval_id to eval_name
        df['model'] = df['model'].str[:15]  # Truncate model names to 15 characters
        df = df.sort_values(by=['pass_percentage'], ascending=[False])  # Sort by pass_percentage descending

        print("\n" + "=" * 50)
        print("Combined Evaluation Summary")
        print("=" * 50)
        print(df.to_string(index=False, header=["Run ID", "Run Name", "Model", "Status", "Pass Percentage (%)", "Error Percentage (%)", "Evaluation ID", "Evaluation Name"]))
        print("=" * 50)

        # Dynamically adjust the figure height based on the number of rows
        num_rows = len(df)
        fig_height = max(3, num_rows * 0.5)  # Set a minimum height of 6 and scale with 0.5 per row


        # Create a horizontal bar chart with rows sorted by pass percentage across all eval_ids
        plt.figure(figsize=(12, fig_height))

        df['display_label'] = df['model'].where(
            (df['model'].str.strip() != '') & (df['model'] != 'None') & (df['model'].notna()),
            df['name']
            )
        
        plt.barh(
            df['display_label'], 
            df['pass_percentage'], 
            color=[eval_id_to_color[eval_id] for eval_id in df['eval_id']], 
            edgecolor='black'
        )
        plt.xlabel('Pass Percentage (%)')
        plt.ylabel('Model')
        plt.title("Pass Percentage by Model Across Evaluations")
        plt.xlim(0, 100)  # Set x-axis scale explicitly to 0-100
        plt.gca().invert_yaxis()  # Invert y-axis to show the highest percentage at the top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Process each run to calculate and collect scores for distribution
        # (This part can be slow as we have to page over results for each run, so we parallelize this.)
        all_scores = []
        run_labels = []
        score_summary = []  # To store data for the summary table

        print("=" * 50)
        print("Fetching scores...")
        print("=" * 50)

        futures = {} # dict of "future: (model, eval_id)" so we can easily access which model powered the run.
        with ThreadPoolExecutor(thread_name_prefix="eval-run-fetcher") as pool:
            for _, row in df.iterrows():
                run_id = row['id']
                eval_id = row['eval_id']
                future = pool.submit(get_eval_run_output_items, client, eval_id, run_id)
                futures.update({ future: (row['model'] , eval_id)})

            for f in as_completed(futures.keys()):
                try:
                    model, eval_id = futures[f]
                    scores = f.result()
                except Exception as e:
                    print(f"exception fetching future result: {e}")
                    scores = None
                if scores:
                    avg_score = sum(scores) / len(scores)
                    min_score = min(scores)
                    max_score = max(scores)
                    p10 = np.percentile(scores, 10)  # 10th percentile
                    p25 = np.percentile(scores, 25)  # 25th percentile
                    p50 = np.percentile(scores, 50)  # 50th percentile (median)
                    p75 = np.percentile(scores, 75)  # 75th percentile
                    p90 = np.percentile(scores, 90)  # 90th percentile

                    # Collect scores and labels for the combined chart
                    all_scores.append((scores, eval_id_to_color[eval_id]))  # Include color for the subplot
                    run_labels.append(f"{model} ({eval_id_to_name[eval_id]})")  # Include eval name in the label

                    # Add data to the summary table
                    score_summary.append({
                        "Model": model,
                        "Evaluation Name": eval_id_to_name[eval_id],
                        "Average Score": f"{avg_score:.2f}",
                        "Min Score": f"{min_score:.2f}",
                        "Max Score": f"{max_score:.2f}",
                        "10th Percentile": f"{p10:.2f}",
                        "25th Percentile": f"{p25:.2f}",
                        "50th Percentile": f"{p50:.2f}",
                        "75th Percentile": f"{p75:.2f}",
                        "90th Percentile": f"{p90:.2f}"
                    })

        # Display the score summary as a table
        if score_summary:
            score_df = pd.DataFrame(score_summary)
            score_df = score_df.sort_values(by=['Evaluation Name', 'Average Score'], ascending=[True, False])  # Sort by eval_name and avg_score
            print("\n" + "=" * 50)
            print("Score Summary Table:")
            print(score_df.to_string(index=False))
            print("=" * 50)

        # Plot all score distributions in a single chart with a maximum of 4 graphs per row
        if all_scores:
            num_runs = len(all_scores)
            max_cols = 4  # Maximum number of graphs per row
            num_rows = (num_runs + max_cols - 1) // max_cols  # Calculate the number of rows

            _, axes = plt.subplots(num_rows, max_cols, figsize=(5 * max_cols, 4 * num_rows), sharey=True)
            axes = axes.flatten()  # Flatten the axes array for easier indexing

            for i, ((scores, color), label) in enumerate(zip(all_scores, run_labels)):
                ax = axes[i]
                ax.hist(scores, bins=10, color=color, edgecolor='black')  # Use color for the histogram
                ax.set_title(label, fontsize=10)  # Include model and evaluation name
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                ax.set_xlim(x_range[0], x_range[1])  # Fix the x-axis range between 0 and 7
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Hide any unused subplots
            for j in range(len(all_scores), len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.suptitle("Score Distributions for each Model", fontsize=16, y=1.02)
            plt.show()
    else:
        print("\n" + "=" * 50)
        print("No evaluation runs found for the provided Evaluation IDs.")
        print("=" * 50)


def get_eval_run_output_items(client: openai.Client, eval_id: str, run_id: str) -> list:
    """
    Fetch the output items for a specific evaluation run and extract the result scores.

    Args:
        eval_id (str): The evaluation ID.
        run_id (str): The run ID.

    Returns:
        list: A list of scores for the output items.
    """
    scores = []

    try:
        response = client.evals.runs.output_items.list(run_id=run_id, eval_id=eval_id)
        for page in response.iter_pages():
            for item in page.data:
                for result in item.results:
                    if result.score is not None:
                        scores.append(result.score)
    except Exception as e:
        print(f"Failed to fetch output items for run {run_id}. Error: {e}")

    return scores
