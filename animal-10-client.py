from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import requests
import io

app = FastAPI()

def predict_via_rest(encoded_image,
                     server_url: str = "http://0.0.0.0:8000/invocations"
                    ) -> np.ndarray:
    headers={"Content-Type": "application/octet-stream"}
    resp = requests.post(server_url, headers=headers, data=encoded_image)
    resp.raise_for_status()
    resp_json = resp.json()

    # Extract the list of predictions
    if "predictions" in resp_json:
        preds_list = resp_json["predictions"]
    elif "instances" in resp_json:
        preds_list = resp_json["instances"]
    else:
        raise ValueError(f"Unexpected response format: {resp_json}")

    preds = np.array(preds_list)
    # If it's 1-D or wrapped oddly, ensure shape is (1, num_classes)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    return preds


translate = {0:'butterfly', 1:'cat', 
            2:'chicken', 3:'cow', 
            4:'dog', 5:'elephant', 
            6:'horse', 7:'sheep', 
            8:'spider', 9:'squirrel'}

@app.post("/upload_file/")
async def create_upload_file(file: UploadFile = File(...)):
    encoded_image = await file.read()  # This reads the file content into bytes
    preds = predict_via_rest(encoded_image)
    result = f"I can see {translate.get(int(preds))} in there"
    return result


if __name__ == "__main__":
    uvicorn.run(
        "animal-10-client:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )