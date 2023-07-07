import pandas as pd
import torch


class ContentBaseMovieStory:
    def __init__(self, model_path, title_path, indices_path):
        """
        Initialize the Content-Based Movie Recommendation system based on movie story.

        Args:
            model_path (str): Path to the model file.
            title_path (str): Path to the title CSV file.
            indices_path (str): Path to the indices CSV file.

        """
        self.indices = None
        self.titles = None
        self.model = None
        self.model_path = model_path
        self.titles_path = title_path
        self.indices_path = indices_path

    def recommendation_story_base(self, movie_name) -> list:
        """
        Generate movie recommendations based on the story of the movies (story-based content recommendation system).

        Args:
            movie_name (str): Name of the movie.

        Returns:
            list: Recommended movies based on the story.

        """
        self.model = torch.load(self.model_path).numpy()
        self.titles = pd.read_csv(self.titles_path)
        self.indices = pd.read_csv(self.indices_path)

        recommendation_movies = self.get_recommendations_story_base(movie_name, self.indices, self.model,
                                                                    self.titles)
        return recommendation_movies

    def get_recommendations_story_base(self, movie_name, indices, cosine_sim, titles, num_recommendations=50):
        """
        Get movie recommendations based on movie story.

        Args:
            movie_name (str): Name of the movie.
            indices (DataFrame): DataFrame containing movie indices.
            cosine_sim (ndarray): Cosine similarity matrix.
            titles (DataFrame): DataFrame containing movie titles.
            num_recommendations (int, optional): Number of recommendations to generate. Default is 50.

        Returns:
            list: Recommended movies based on the movie story.

        """
        idx = indices[indices['title'] == movie_name].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]['title'].values


if __name__ == "__main__":
    model_path = '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/Overview Model/cosine_sim.pkl'
    titles_path = '/Recommender_System/master_ui/Models/Overview Model/overview_titles.csv'
    indices_path = '/Recommender_System/master_ui/Models/Overview Model/overview_indices.csv'
    test = ContentBaseMovieStory(model_path, titles_path, indices_path)
    movie_name = "Super High Me"
    recommendation = test.recommendation_story_base(movie_name)
    print(recommendation)
    print(len(recommendation))
