from fastapi import FastAPI, File, Response, UploadFile, HTTPException
from roboflow import Roboflow
import os
import subprocess
from inference import InferencePipeline
from roboflow import Roboflow
from inference.core.interfaces.stream.sinks import VideoFileSink

app = FastAPI()

# Initialize the inference pipeline
model_id = "merged-project-2/1"
output_file_name = "outputfile.mp4"


 
@app.post("/process_video")
async def process_video(video_file: UploadFile = File(...)):
    # Save the uploaded video to a local file
    with open("uploaded_video.mp4", "wb") as buffer:
        buffer.write(await video_file.read())

    # Initialize VideoFileSink
    video_sink = VideoFileSink.init(video_file_name="outputfile.avi")

    # Initialize and start the inference pipeline
    pipeline = InferencePipeline.init(
        model_id=model_id,
        api_key= "INc4g2WbMuzVOyCAXNVp",
        video_reference="uploaded_video.mp4",
        on_prediction=video_sink.on_prediction
    )
    pipeline.start()
    pipeline.join()
    video_sink.release()

    # Convert outputfile.avi to outputfile.mp4
    conversion_command = ["ffmpeg", "-y", "-i", "outputfile.avi", "outputfile.mp4"]
    subprocess.run(conversion_command)

    # Read the processed video file
    with open("outputfile.mp4", "rb") as video_file:
        video_data = video_file.read()

    # Clean up local files
    os.remove("outputfile.avi")
    os.remove("outputfile.mp4")
    os.remove("uploaded_video.mp4")

    # Return the processed video and results as response
    return Response(content=video_data, media_type="video/mp4")



rf = Roboflow(api_key="INc4g2WbMuzVOyCAXNVp")
project = rf.workspace().project("merged-project-2")
model = project.version("1").model

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    # Save the uploaded video file
    with open("uploaded_video.mp4", "wb") as buffer:
        buffer.write(await file.read())

    # Perform prediction on the uploaded video
    job_id, signed_url, expire_time = model.predict_video(
        "uploaded_video.mp4",
        fps=5,
        prediction_type="batch-video",
    )

    # Poll until results are ready
    results = model.poll_until_video_results(job_id)

    os.remove("uploaded_video.mp4");

    if not results:
        raise HTTPException(status_code=404, detail="Video prediction results not found")

    return results