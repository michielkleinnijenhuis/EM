#
#     if plotpairs:
#         full1, full2 = downsample_images(p, imgs, 1)
#         kp1 = np.c_[kp1, np.ones(kp1.shape[0])]
#         kp2 = np.c_[kp2, np.ones(kp2.shape[0])]
#         kpfull1 = kp1.dot(S)[:,:2]
#         kpfull2 = kp2.dot(S)[:,:2]
#         plot_pair_ransac(outputdir, 'f' + pairstring, p, full1, full2, kpfull1, kpfull2, matches, inliers)
