# ArtOa
 AI project for ArtOa

##### Start with:
1. git clone (repo)
2. cd (repo)
3. pip install virtualenv
4. virtualenv venv
5. source venv/bin/activate
6. pip install -r requirements.txt
7. add next files to pipeline folder remover_v1.pth, maskrcnn_v2.pth
8. create .env file in pipeline folder in which you have to write down your openai API key: OPENAI_API_KEY=""
9. add empty static folder to pipeline
10. add pretrained folder to pipeline
11. in terminal go to pipeline folder: cd pipeline
12. type <uvicorn app:app --reload> for debugging mode or <uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4> for production (define host, port and number of workers by you preferences
