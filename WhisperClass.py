import whisper

model = whisper.load_model("medium")
result = model.transcribe("Ganga.m4a")
print(result)