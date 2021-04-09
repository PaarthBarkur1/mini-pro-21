from flask import Flask,render_template,request,url_for,redirect
from classification import *
app= Flask(__name__)

@app.route("/", methods=["POST","GET"])
def list():
    if request.method=="POST":
        model=request.form["model"]
        var_smoothing=request.form["var_smoothing"]
        n_neighbors=request.form['n_neighbors']
        leaf_size=request.form['leaf_size']
        max_depth=request.form['max_depth']
        min_samples_split=request.form['min_samples_split']
        n_estimators=request.form['n_estimators']
        random_state=request.form['random_state']
        max_leaf_nodes=request.form['max_leaf_nodes']
        
        
        output(model,var_smoothing,n_neighbors,leaf_size,max_depth,min_samples_split,n_estimators,random_state,max_leaf_nodes)
        return render_template('form.html')
    else:
        return render_template("model.html")

if __name__ == "__main__":
    app.run()