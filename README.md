The "Board Game Analysis" project aims to explore and analyze data from BoardGameGeek (BGG), one of the largest online communities for board game enthusiasts. This project will delve into the characteristics of board games and their relationships with user ratings, providing insights into what makes a game highly rated and popular.


**Important Notes**

*Non-Commercial Use Only*: This package is intended for non-commercial use only due to the terms of the BoardGameGeek XML API. Any commercial use requires a separate license from BoardGameGeek. More information [here](https://boardgamegeek.com/wiki/page/XML_API_Terms_of_Use#).

*Attribution*: This package uses data from BoardGameGeek. We acknowledge and appreciate their contribution to the board gaming community. Please ensure that any public-facing use of this package includes attribution to BoardGameGeek as the source of the data and includes the BoardGameGeek logo linked back to https://boardgamegeek.com.


## Start 🔌 API & 🖥️ GUI *locally* (👉 development)

### Setup (only once)
You need to install Uvicorn if you're running it for the first time.
This is done via the *Make* file *reinstall_package*:
```shell
pip install uvicorn
```

Than you can call the REST API service (FastAPI) and the GUI service (Streamlit)

Start the **FastAPI** app in your terminal
```shell
uvicorn package_folder.api_file:app --reload
```

Start the **Streamlit** app in your terminal
```shell
streamlit run streamlit/app.py   
```

## Start the 🔌 API container image in Docker or GCP

### Start containter local into Docker
Create a `.env` file with parameters of your GC:  

An example :
```properties
IMAGE=api_bgg
GCP_REGION=europe-west1
ARTIFACTSREPO=api-bgg
# Replace this with your GCP project name.
GCP_PROJECT=[Name_of_our_GCP_project]
MEMORY=2Gi
```
For a local docker run, use the following `Make` commands:

```shell
make build_container_local
```
```shell
make run_container_local
```

### Setup artifactory repository on GCP (⚠️ run only once)
The following commands you have to run only one timer per GCP

```shell
make allow_docker_push
```
```shell
make create_artefacts_repo
```

### Create new image on GCP
For every creation of a new container version on GCP, you have to run the following commands.
These commands depend on your OS.

* Windows user (build image)
```shell
    make build_for_production
```

* Mac user (build image)
```shell
    make m_chip_build_image_production
```

... and the **container image push** and **deploy** commands are the same on both platforms

```shell
  make push_image_production
```

```shell
  make deploy_to_cloud_run
```

After the deployment command, you can access the GCP version with the link like this e.g.: `https://apibgg-{name of our GCP}.europe-west1.run.app/docs`
![GCP-doc](/docs/images/gcp-doc.png)

## Start 🖥️ GUI Streamlit App in the Cloud
To launch the Streamlit app in the Streamlit Cloud, follow these steps:
* Ensure you have a running GCP API with the imported ML model.
* **Update the URI** in the Streamlit app's source code to point to the GCP API.
* Have admin access to the GitHub repository of the Streamlit application.
  If you don't have the necessary rights, you can fork the project on GitHub.
  (Fork either all branches or just the master branch, depending on your needs.)  
  ⚠️ Important: Use the forked repository solely for Streamlit deployment purposes.
  Do not modify or commit code changes to this repository.
* 🏁 Create the Streamlit application using the link to your forked repository and selected branch. 🏁

### Create the Streamlit App
1. If you're not the admin of the repository
   (you are only a teammate — and not god 😇 or the other kind 😈) create a
fork on Github ![Fork-Github](/docs/images/fork_github.png)

2. Open the [Streamlit Cloud](https://streamlit.io/cloud) service and "Join Community Cloud"
3. Create an app (right upper bottom) and choose **GitHub** as a source
4. Add the value: 
    * Your own repository (You must give Streamlit Cloud access before)
    * The branch you want to deploy from
    * Location of the Streamlit app. ⚠️ Pay attention to whether the application is in a subfolder.
    * Leave the App URL as it is.
    * 🚨 Choose the Python version of 3.10 in the "Advanced settings"  
    ![Streamlit](/docs/images/streamlit_new.png)
5. Then deploy and enjoy the baked goods 🥐



