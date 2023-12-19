FROM python:3.9
WORKDIR /app
# COPY mymodel.py
COPY Training_model/classes classes/
# COPY docker_inference.py
COPY Inference_model/docker_inference.py .
# COPY class_labels
COPY Classes_description/class_labels data/class_labels
# COPY sign_names.csv
COPY Classes_description/sign_names.csv data/
# COPY Low_quality.png
COPY readme_files/Low_quality.png data/
# COPY segmented_images
COPY Data/segmented_images data/segmented_images
# COPY segmented_signs/model_32
COPY Data/segmented_signs/model_32 data/segmented_signs/model_32
# COPY model_43M_32x32_9966.hdf5
COPY Training_model/models_weights/model_43M_32x32_9966.hdf5 weights/
# COPY docker_requirements.txt
COPY ./docker_requirements.txt ./
RUN pip install --no-cache-dir -r docker_requirements.txt
CMD ["python", "./docker_inference.py"]
