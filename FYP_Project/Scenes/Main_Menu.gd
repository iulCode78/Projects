extends Control

#When pressed, the game resumes
func _on_play_pressed():
	get_tree().change_scene_to_file("res://Scenes/Level_1.tscn")

#When pressed, you quit to the main menu screen
func _on_quit_pressed():
	get_tree().quit()

#When pressed, you quit to the desktop
func _on_options_pressed():
	get_tree().change_scene_to_file("res://Scenes/Options.tscn")
