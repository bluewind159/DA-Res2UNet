from lightweight_gan.cli import train_from_folder
import fire
def main():
    fire.Fire(train_from_folder)
    
if __name__ == '__main__':
    main()