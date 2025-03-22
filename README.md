The "Board Game Analysis" project aims to explore and analyze data from BoardGameGeek (BGG), one of the largest online communities for board game enthusiasts. This project will delve into the characteristics of board games and their relationships with user ratings, providing insights into what makes a game highly rated and popular.


**Important Notes**

*Non-Commercial Use Only*: This package is intended for non-commercial use only due to the terms of the BoardGameGeek XML API. Any commercial use requires a separate license from BoardGameGeek. More information [here](https://boardgamegeek.com/wiki/page/XML_API_Terms_of_Use#).

*Attribution*: This package uses data from BoardGameGeek. We acknowledge and appreciate their contribution to the board gaming community. Please ensure that any public-facing use of this package includes attribution to BoardGameGeek as the source of the data and includes the BoardGameGeek logo linked back to https://boardgamegeek.com.


## Start API locally (in development)
You need to install Uvicorn if you're running it for the first time:
```shell
pip install uvicorn
```

Then you can start the API from within the `bgg-project` folder:

```shell
uvicorn api_file:app --reload
```

## Start the API container image in Docker or GCP

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
