from flask import Flask, render_template, request
import joblib

app=Flask(__name__)

model=joblib.load('dib_79.pkl')

@app.route('/')
def welcome():
    return render_template('wc.html')

@app.route('/home', methods= ['POST'])
def homepage():
    return render_template('home.html')

@app.route('/result', methods= ['POST'])
def contact():
    preg=request.form.get('age')
    plas=request.form.get('plas')
    pres=request.form.get('pres')
    skin=request.form.get('skin')
    test=request.form.get('test')
    mass=request.form.get('mass')
    pedi=request.form.get('pedi')
    age=request.form.get('age')

    print(int(preg) ,int(plas), int(pres), int(skin), int(test), int(mass), int(pedi), int(age))

    pred=model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])
    if pred[0]==1:
        op='diabetic'
    else:
        op='not diabetic'   
    return render_template('result.html', predicted_text=f'The person is {op}')
#run the app
if __name__ == '__main__':
    app.run(debug=True)
