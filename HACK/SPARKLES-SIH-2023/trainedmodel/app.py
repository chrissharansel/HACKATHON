

from flask import Flask, render_template, request, jsonify
import instaloader
import numpy as np
import tensorflow as tf

app = Flask(__name__)

def get_instagram_data(username):
    # ... (same function as in your provided script)
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
            "userFollowerCount": profile.followers,
            "userFollowingCount": profile.followees,
            "userBiographyLength": len(profile.biography),
            "userMediaCount": profile.mediacount,
            "userHasProfilPic": int(not profile.is_private and profile.profile_pic_url is not None),
            "userIsPrivate": int(profile.is_private),
            "usernameDigitCount": sum(c.isdigit() for c in profile.username),
            "usernameLength": len(profile.username),
        }
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile with username '{username}' not found.")
        return None
# ...
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the username from the request
        username = request.json['username']

        # Get Instagram data
        insta_data = get_instagram_data(username)

        if insta_data:
            # Load the trained model
            load_model = tf.keras.models.load_model('trainedmodel')

            # Convert Instagram data to NumPy array
            X_new = np.array([list(insta_data.values())], dtype=np.float32)

            # Make predictions
            predictions = load_model.predict(X_new)

            # Return the result as JSON
            result = {
                "result": f"Prediction for {username}: {'Fake' if predictions[0][0] >= 0.5 else 'Real'} (Probability: {predictions[0][0]:.4f})"
            }
            return jsonify(result)
        else:
            return jsonify({"result": f"Profile with username '{username}' not found."})
    except Exception as e:
        return jsonify({"result": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
