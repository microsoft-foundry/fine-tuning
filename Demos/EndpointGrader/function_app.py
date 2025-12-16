from typing import Any, Dict
import json
import logging

import azure.functions as func

from grader import grade


# Initialize our Azure Function app. We use key-based auth.
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="grader", methods=["POST"])
def grader(req: func.HttpRequest) -> func.HttpResponse:
    """
    Grade a model response. Same API as a Python grader, meaning you get:

      sample -- the output from the model
      item -- the row of training/validation data used to generate sample

    Must return a JSON object with a "score" member that's a JSON number.
    """
    try:
        data: Dict[str, Any] = req.get_json()
        sample = data.get("sample", {})
        item = data.get("item", {})

        # log things for debugging for now
        logging.info(f"sample: {sample}")
        logging.info(f"item: {item}")

        # call our grader
        score = grade(sample, item)

        logging.info(f"resulting score {score}")
        return func.HttpResponse(
            json.dumps({"score": score}),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": {"message": str(e)}}),
            mimetype="application/json",
            status_code=500,
        )
