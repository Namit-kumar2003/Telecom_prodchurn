import json

def load_data():
    try:
        with open('youtube.txt', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    
def save_data_helper(videos):
    with open('youtube.txt', 'w') as file:
        json.dump(videos, file)


def list_all_videos(videos):
    print("\n")   
    print('*'*50)
    for index, videos in enumerate(videos, start=1):
        print(f"{index}.{videos['name']}, Duration: {videos['length']} ")
    print("\n")
    print('*'*50)

def add_video(videos):
    name = input("Enter the name of the video: ")
    length = input("Enter the length of the video: ")
    videos.append({'name': name, 'length': length})
    save_data_helper(videos)

def update_video(videos):
    list_all_videos(videos)
    index = int(input("Enter the video number you want to update: "))
    if 1<= index <= len(videos):
        name = input("Enter the new video name: ")
        length = input("Enter the new video length: ")
        videos[index-1] = {'Name': name, 'Length': length}
        save_data_helper(videos)
    else:
        print("Invalid index selected!!!")

def delete_video(videos):
    list_all_videos(videos)
    index = int(input("Enter the video number you want to delete: "))
    if 1<= index <= len(videos):
        del videos[index-1]
        save_data_helper(videos)
    else:
        print("Invalid choice entered!!!")




def main():
    videos = load_data()
    while True:
        print("\n Youtube Manager | Choose an option")
        print("1. List all youtube videos")
        print("2. Add youtube videos")
        print("3. Update youtube videos")
        print("4. Delete youtube video")
        print("5. Exit the App")
        choice = input("Enter your choice: ")
        print(videos)

        match choice:
            case '1':
                list_all_videos(videos)
            case '2':
                add_video(videos)
            case '3':
                update_video(videos)
            case '4':
                delete_video(videos)
            case '5':
                break 
            case _:
                print("Invalid choice!!!")
    
if __name__ == "__main__":
    main()