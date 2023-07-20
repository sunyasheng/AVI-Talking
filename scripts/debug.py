import glob
import os


def main():
    output_root = '/data/yashengsun/local_storage/Mead_emoca/Mead_W'
    paths = os.listdir(output_root)
    # print(paths)

    new_paths = [os.path.join(os.path.dirname(p), 'W'+os.path.basename(p)[1:]) for p in paths]
    for path, new_path in zip(paths, new_paths):
        # print(path, new_path)
        cmd = 'mv {} {}'.format(os.path.join(output_root,path, ('M'+ path[1:]).split('.')[0]+'.wav'),
                                os.path.join(output_root,new_path,new_path.split('.')[0]+'.wav'))
        print(cmd)
        os.system(cmd)
        # break


if __name__ == '__main__':
    main()
