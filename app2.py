from flask import Flask,render_template,request,url_for,redirect
from classification import *
app= Flask(__name__)

@app.route("/", methods=["POST","GET"])
def list():
    if request.method=="POST":
        model=request.form["model"]
        alpha=request.form["alpha"]
        if type(alpha)!=float:
            alpha=1
        n_neighbors=(request.form['n_neighbors'])
        if type(n_neighbors)!=int:
            n_neighbors=5
        leaf_size=request.form['leaf_size']
        if type(leaf_size)!=int:
            leaf_size=30
        max_depth=(request.form['max_depth'])
        if type(max_depth)!=int:
            max_depth=50
        min_samples_split=request.form['min_samples_split']
        if type(min_samples_split)!=int:
            min_samples_split=2
        n_estimators=(request.form['n_estimators'])
        if type(n_estimators)!=int:
            n_estimators=500
        random_state=request.form['random_state']
        if type(random_state)!=int:
            random_state=42
        max_leaf_nodes=request.form['max_leaf_nodes']
        if type(max_leaf_nodes)!=int:
            max_leaf_nodes=50
        
        output(model,float(alpha),int(n_neighbors),int(leaf_size),int(max_depth),int(min_samples_split),int(n_estimators),int(random_state),int(max_leaf_nodes))
        return render_template("testing.html")
    else:
        return render_template("form.html")

if __name__ == "__main__":
    app.run()