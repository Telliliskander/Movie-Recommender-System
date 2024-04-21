from flask import Flask, render_template, request
import model 

app = Flask(__name__)

MOVIE_DATA = {
    "movie_1.jpg": 5,
    "movie_2.jpg": 5,
    "movie_3.jpg": 0.5,
}

title_to_path={'School of Rock (2003)':'School of Rock (2003).jpg',
                       'Producers, The (2005)':'the producers (2005).jpg', 
                       'Mystery Men (1999)':'mystery men (1999).jpg',
                       'Mad Hot Ballroom (2005)':'Mad Hot Ballroom (2005).jpg',
                       'Fall, The (2006)':'the fall (2006).jpg',
        }

@app.route("/")
def index():
    return render_template("results.html", movies=MOVIE_DATA)


data = {}

@app.route('/rate_movies', methods=['GET', 'POST'])
def rate_movies():
    if request.method == 'POST':
        # Get the ratings from the form
        ratings = {
            'Dumb & Dumber (Dumb and Dumber) (1994)': float(request.form['rating1']),
            'Home Alone (1990)': float(request.form['rating2']),
            'Titanic (1997)': float(request.form['rating3']),
            'Call Me by Your Name (2017)': float(request.form['rating4']),
            'Hangover, The (2009)': float(request.form['rating5']),
        }


        disp=model.predict(ratings)
        
        # fun that creates dict that maps the predicted movies to their posters
        final_dic={}
        for key,val in disp.items():
            final_dic[key]=title_to_path[key]

        # Update the global data variable
        data['ratings'] = disp
           

        return render_template("results.html",disp=disp, movies=final_dic)

    else:
        return render_template('index.html')
    



if __name__ == "__main__":
    app.run(debug=True)

