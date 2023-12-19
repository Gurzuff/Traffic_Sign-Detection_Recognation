FROM python:3.9
WORKDIR /app
# COPY mymodel.py
COPY Training_model/classes app/
# COPY class_labels
COPY Classes_description/class_labels app/data/
# COPY sign_names.csv
COPY Classes_description/sign_names.csv app/data/
# COPY Low_quality.png
COPY readme_files/Low_quality.png app/data/
# COPY segmented_images
COPY Data/segmented_images app/data/
# COPY segmented_signs/model_32
COPY Data/segmented_signs/model_32 app/data/segmented_signs
# COPY model_43M_32x32_9966.hdf5
COPY Training_model/models_weights/model_43M_32x32_9966.hdf5 app/weights/
RUN pip install --no-cache-dir -r Docker/docker_requirements.txt
CMD ["python", "Docker/docker_inference.py"]
