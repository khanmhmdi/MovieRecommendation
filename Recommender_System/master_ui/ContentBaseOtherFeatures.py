import pandas as pd
import torch


class ContentBaseOtherFeatures:
    def __init__(self, model_path, data_path, indices_path):
        """
        Initialize the Content-Based Movie Recommendation system based on other features.

        Args:
            model_path (str): Path to the model file.
            data_path (str): Path to the data CSV file.
            indices_path (str): Path to the indices CSV file.

        """
        self.indices = None
        self.data = None
        self.model = None
        self.model_path = model_path
        self.data_path = data_path
        self.indices_path = indices_path

    def recommendation_genre_cast_keywords_crew(self, movie_name) -> list:
        """
        Generate movie recommendations based on the story of the movies (story-based content recommendation system).

        Args:
            movie_name (str): Name of the movie.

        Returns:
            list: Recommended movies based on the story.

        """
        self.model = torch.load(self.model_path).numpy()
        self.data = pd.read_csv(self.data_path)
        self.indices = pd.read_csv(self.indices_path)

        recommendation_movies = self.get_recommendations_genre_cast_keywords_crew_base(movie_name, self.indices,
                                                                                       self.model,
                                                                                       self.data)
        return recommendation_movies

    def get_recommendations_genre_cast_keywords_crew_base(self, Movie_name, indices, cosine_sim, data,
                                                          num_recommendations=20):
        """
        Get movie recommendations based on genre, cast, keywords, and crew.

        Args:
            movie_name (str): Name of the movie.
            indices (DataFrame): DataFrame containing movie indices.
            cosine_sim (ndarray): Cosine similarity matrix.
            data (DataFrame): DataFrame containing movie data.
            num_recommendations (int, optional): Number of recommendations to generate. Default is 50.

        Returns:
            list: Recommended movies based on genre, cast, keywords, and crew.

        """
        print(Movie_name)
        idx = indices[indices['title'] == Movie_name].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations]
        movie_indices = [i[0] for i in sim_scores]
        movie_scores = [i[1] for i in sim_scores]
        movies = data.iloc[movie_indices]['title']
        print("get_recommendations_genre_cast_keywords_crew_base results: ")
        print(pd.DataFrame({'Recommended Movie': movies, 'Similarity Score': movie_scores}))
        return movies.values


if __name__ == "__main__":
    model_path = '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/genres_cast_keywords_crew Model/cosine_sim.pkl'
    data_path = '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/genres_cast_keywords_crew Model/data.csv'
    indices_path = '/home/mkhanmhmdi/Desktop/ML Project/MovieRecommendation/Recommender_System/master_ui/Models/genres_cast_keywords_crew Model/indices.csv'
    test = ContentBaseOtherFeatures(model_path, data_path, indices_path)
    movie_name = 'Mala Noche'
    recommendation = test.recommendation_genre_cast_keywords_crew(movie_name)
    print(recommendation)
    print(len(recommendation))
