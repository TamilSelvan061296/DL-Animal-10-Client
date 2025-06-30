from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import requests
import io
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Animal-10 Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f4f4; padding: 40px; }
            .container { background: #fff; padding: 30px 40px; border-radius: 8px; max-width: 400px; margin: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
            h2 { text-align: center; }
            #preview { display: block; margin: 20px auto; max-width: 100%; max-height: 200px; border-radius: 6px; }
            #result { margin-top: 20px; text-align: center; font-size: 1.2em; }
            .btn { display: block; width: 100%; padding: 10px; background: #007bff; color: #fff; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; }
            .btn:disabled { background: #aaa; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Animal-10 Classifier</h2>
            <input type="file" id="fileInput" accept="image/*"><br>
            <img id="preview" src="#" alt="Image preview..." style="display:none;">
            <button class="btn" id="uploadBtn" disabled>Upload & Classify</button>
            <div id="result"></div>
        </div>
        <script>
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const uploadBtn = document.getElementById('uploadBtn');
            const resultDiv = document.getElementById('result');
            let selectedFile = null;

            fileInput.addEventListener('change', function() {
                const file = this.files[0];
                if (file) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                    uploadBtn.disabled = false;
                    resultDiv.textContent = '';
                } else {
                    preview.style.display = 'none';
                    uploadBtn.disabled = true;
                }
            });

            uploadBtn.addEventListener('click', function() {
                if (!selectedFile) return;
                uploadBtn.disabled = true;
                resultDiv.textContent = 'Classifying...';
                const formData = new FormData();
                formData.append('file', selectedFile);

                fetch('/upload_file/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    resultDiv.textContent = data;
                })
                .catch(error => {
                    resultDiv.textContent = 'Error: ' + error;
                })
                .finally(() => {
                    uploadBtn.disabled = false;
                });
            });
        </script>
    </body>
    </html>
    """
    return content


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