"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = [
    "mcastrolab/Brazil-Covid19-e0-change",
    "jschoeley/de0anim",
    "sychi77/Thoracic_Surgery_Patient_Survival",
    "ashtad63/HackerRank-Data-Scientist-Hiring-Test",
    "OxfordDemSci/ex2020",
    "hemanth/life-expectancy",
    "libre-money-projects/Geconomicus",
    "michaelstepner/healthinequality-code",
    "vkontis/maple",
    "Public-Health-Scotland/Life-Expectancy",
    "eliasdabbas/life_expectancy",
    "yc244/Stata-code_BMC-Medince2019_Chudasama",
    "BatuhanSeremet/Life_Expectancy-Regression",
    "hisham-maged10/Live-Data-Map-Visualization-Application",
    "mpascariu/lemur",
    "trent129/Life-Expectancy-Prediction",
    "aleynahukmet/life-expectancy-prediction",
    "rinaldoclemente/Life-Expectancy-Dataset-Analysis",
    "philngo/mormon-prophets",
    "aris-plexi/GDP_LE",
    "basnetbik/SocietyRecommenderSystem",
    "zahta/Covid-19-Project",
    "datasets/london-life-expectancy",
    "SmartPracticeschool/llSPS-INT-1942-Predicting-Life-Expectancy-using-Machine-Learning",
    "open-numbers/ddf--gapminder--life_expectancy",
    "alisaghilutfi/Life-Expectancy-Prediction",
    "ofunkey/Life-Expectancy-Visualization",
    "emmanuelAkpe/LifeExpectancy",
    "patrickaubert/healthexpectancies",
    "AbeyK/LifeExpectancyCalculator",
    "Moostafa246/LifeExpectancyPythonProject",
    "LindsayBauer/LifeExpectancyAnalysis",
    "KodPlanet/life_expectancy",
    "abhijeet65/life_expectancy",
    "JoshuaDaleYoung/Life-Expectancy",
    "MagnusHambleton/life-expectancy",
    "JosephLazarus/Life_Expectancy",
    "francisatoyebi/Predicting-Life-Expectancy---WHO",
    "Fraddle9/ML-Life-Expectancy",
    "shubh0125/Life_Expectancy_Analysis",
    "osmanerikci/life_expectancy_app",
    "euroargodev/Argo_life_expectancy_analyses",
    "chen-gloria/life-expectancy-calculator",
    "VideshLoya/Predicting-Life-Expectancy",
    "chessvision-ai/chess-pieces-lifetime-expectancy",
    "alessandra-barbosa/life_expectancy_prediction",
    "NEXTSLIM/Life-Expectancy-Web",
    '''
    "sagarsharma786/SmartPracticeschool-llSPS-INT-1694-Predicting-Life-Expectancy-using-Machine-Learning",
    "mathchi/DS_the-Life-Expectancy-Data",
    "taylor2325/country_GDP_and_life_expectancy_Compared",
    "alexsmartens/d3-chart_income-vs-life-expectancy”,
    "helenaEH/Scatterplot_animation”,
    "Junyong-Suh/D3MultiGraph”,
    "merijnbeeksma/predict-EoL”,
    "shivangi2012/Multiple-Linear-Regression-Analysis”,
    "karthickganesan/Life-Expectacny-Prediction”,
    "apj2n4/Exploratory-Analysis-GDP-Life-Electricity”,
    "yc244/Stata-code_Journal-of-Internal-Medicine-2019_Chudasama”,
    "usamabilal/SALURBAL_MS10”,
    "wrigleyfield/hypotheticalwhite2020mortality”,
    "AnshBhatti/COVID-19_Data_Analysis”,
    "timriffe/PHDS_HLE_2021”,
    "sharmaroshan/Europe-Political-Data-Analysis”,
    "thegrigorian/WBCrossCountryTimeSeries”,
    "kanishkan91/R-shiny-animation-applications”,
    "elcarpins/Li-ionSOCalgorithm_gradientBoosting”,
    "HamoyeHQ/17-life-expectancy”,
    "t-redactyl/life-expectancy-analysis”,
    "SAILResearch/clone-life-expectancy”,
    "PPgp/bayesLife”,
    "pjmerica/415-final-Project
    "SmartPracticeschool/llSPS-INT-1983-Predicting-Life-Expectancy-using-Machine-Learning”,
    "SmartPracticeschool/llSPS-INT-1644-Predicting-Life-Expectancy-using-Machine-Learning”,
    "SmartPracticeschool/llSPS-INT-2004-Predicting-Life-Expectancy-using-Machine-Learning”,
    "hammad93/astropreneurship-hackathon-2019”,
    "r2ressler/life_exp_r2ressler”,
    "bneiluj/expectancy”,
    "vezraabah/Factors-Affecting-Life-Expectancy”,
    "neilnatarajan/toilet_used”,
    "AntoineGuiot/OWKIN_PROJECT”,
    "agbarnett/pollies”,
    "karthik-panambur/LifeExpectancy”,
    "MingXXI/LifeExpectancyLinearModel”,
    "SamruddhiGoswami/LifeExpectancyAnalysis”,
    "SunpreetSChahal/LifeExpectancyAnalysis”,
    "YoungAndJin96/Regression_LifeExpectancy”,
    "FreshOats/LifeExpectancy_v_GDP”,
    "Abhishek7884/Life1expectancy”,
    "ptyadana/DV-Data-Visualization-with-Python”,
    "krinab98/krinab98.github.io”,
    "kirillwolkow/Data-Analysis-Life-GDP”,
    "Priyanka11-art/Life-Expectancy-Post-Thoracic-Surgery”,
    "igordpm/life-expectancy”,
    "Pradhuman099/Predicting-Life-Expectancy-”,
    "sejaldua/life-expectancy”,
    "subhajournal/Life-Expectancy”,
    "NEXTSLIM/Life-Expectancy”,
    "mohanakamanooru/Life-Expectancy”,
    "muoten/life-expectancy”,
    "Abhinav1092/Life-expectancy”,
    "tanny459/Lifer-Expectancy”, 
    "leahdassler/Life_Expectancy”,
    "cnahmetcn/life_expectancy”,
    "tgoel5884/Life-Expectancy”,
    "jmt0221/Life_Expectancy”,
    "nmshafie1993/Life_Expectancy”, 
    "mrrehani/Life-Expectancy”, 
    "siddharth-star/Life-Expectancy”, 
    "JeremyMiranda/Life-Expectancy”,
    "SkyThonk/ML-Life-Epectancy-Prediction”,
    '''
]

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)