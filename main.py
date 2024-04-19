from fastapi import FastAPI, File, UploadFile, HTTPException
from roboflow import Roboflow

app = FastAPI()

rf = Roboflow(api_key="INc4g2WbMuzVOyCAXNVp")
project = rf.workspace().project("blackpod_cocoa")
model = project.version("2").model

@app.post("/predict_video/")
async def predict_video(video_file: UploadFile = File(...)):

        # Save the uploaded video file
        with open("uploaded_video.mp4", "wb") as buffer:
         buffer.write(await video_file.read())
        
        # Predict on the uploaded video file
        job_id, signed_url, expire_time = model.predict_video(
            "uploaded_video.mp4",
            fps=5,
            prediction_type="batch-video"
        )
        
        # Poll until the prediction job is completed
        results = model.poll_until_video_results(job_id)
        
        # Return the prediction results
        return {"results": results}
    
   
