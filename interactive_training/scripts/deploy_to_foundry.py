"""Create or delete a Fireworks deployment in Foundry."""

import argparse

from interactive_training.deployment import DeploymentTarget, create_deployment, delete_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "project_id",
        help=(
            "Full Foundry project ARM ID:\n  /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/"
            "<RESOURCE_GROUP>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/"
            "projects/<PROJECT_NAME>"
        ),
    )
    parser.add_argument("session_id", nargs="?", help="Existing session_ or model_ ID (create only)")
    parser.add_argument("--target", choices=[target.value for target in DeploymentTarget], default="Fireworks")
    parser.add_argument("--sku-capacity", type=int, default=50, help="Provisioned SKU capacity (default: 50)")
    parser.add_argument("--delete", metavar="DEPLOYMENT_NAME", help="Delete this deployment instead of creating")
    args = parser.parse_args()
    if args.delete:
        if args.session_id:
            parser.error("session_id must be omitted with --delete")
        delete_deployment(args.project_id, args.delete)
        print(f"Deployment deleted: {args.delete}")
    else:
        if not args.session_id:
            parser.error("session_id is required unless --delete is specified")
        deployment = create_deployment(
            args.project_id, args.session_id, DeploymentTarget(args.target), args.sku_capacity
        )
        print(f"Deployment succeeded: {deployment['name']}")


if __name__ == "__main__":
    main()
