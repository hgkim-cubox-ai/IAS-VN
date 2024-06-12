def infer(args, model, loader):    
    
    model.eval()
    
    z_dict = {}
    for i in range(10):
        z_dict[i] = []
    
    with tqdm(loader, unit='batch') as bar:
            for i, (img, label) in enumerate(bar):
                if i == 500:
                    break
                img = img.cuda()
                label = label.cuda()
                img_hat, z = model(img)
                
                for n in range(img.size(0)):
                    i = img[n, 0].detach().cpu().numpy()
                    i_hat = img_hat[n, 0].detach().cpu().numpy()
                    
                    z_dict[label.item()].append(z.detach().cpu().numpy().squeeze())
                    
                    # print(label)
                    # cv2.imshow('i', np.hstack([i, i_hat]))
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    
    
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(k, cmap='gray')
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(i_hat, cmap='gray')
                    # plt.axis('off')
                    # plt.show()
    
    print('')