FROM continuumio/miniconda3

# Set up user.
ARG user_name="clvusr"
RUN groupadd --gid 5000 $user_name && \
    useradd --home-dir /home/$user_name --create-home --uid 5000 \
    --gid 5000 $user_name

# Set working directory.
WORKDIR /home/$user_name/clairvoyance

# Set up the conda environment.
ARG conda_env="clvenv"
COPY ./environment.yml ./environment.yml
RUN conda update -n base conda && \
    conda env create --name $conda_env -f ./environment.yml && \
    echo "conda activate $conda_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

# Copy everything.
COPY --chown=$user_name:$user_name . .

# Run environment check.
USER $user_name
