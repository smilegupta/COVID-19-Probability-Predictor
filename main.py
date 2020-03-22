from flask import Flask,render_template,request
import pickle
app = Flask(__name__)

file=open('model.pk1','rb')
clf=pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        print(request.form)
        myDict = request.form
        fever = float(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        difffBreath = int(myDict['difffBreath'])
        inputFeatures = [fever,pain,age,runnyNose,difffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html',inf=round(infProb*100))
    return render_template('index.html')
    # return 'Hello, World'+" "+str(infProb)

if __name__ == "__main__":
    app.run(debug=True) 