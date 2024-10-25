
import pygame
import time


def play_ending_song(addr:str = '../data/Sinfonia To Cantata # 29.mp3')-> None:
    """
    Function to play a song when the program ends

    Parameters:
    addr (str): Address of the MP3 file

    Returns:
    None
    
    
    """
    # Initialize the mixer
    pygame.mixer.init()
    # Load the MP3 file
    pygame.mixer.music.load(addr)
    # Play the MP3 file
    pygame.mixer.music.play()


def stop_ending_song(seconds:int = 5)->None:
    """
    Function to stop the song that is playing

    Parameters:
    seconds (int): Number of seconds to wait before stopping the song

    Returns:
    None
    """

    time.sleep(seconds)
    pygame.mixer.music.stop()
    # Optional: Clean up the mixer
    pygame.mixer.quit()