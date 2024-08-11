def plot_results(system,time_plot,time_plot_sequential,output,network_sol,list_of_labels,n_x,L,vec,avg_time=-1,number_iterates=1,btype=None,network_sol_lobatto=[],avg_time_lobatto=None,network_sol_SIR_flow=[],avg_time_SIR_flow=None):

    import numpy as np
    import matplotlib.pyplot as plt
    import time as time_lib
    import os
    import matplotlib
    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D

    #Setting the plotting parameters
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    matplotlib.rcParams['font.size']= 45
    matplotlib.rcParams['font.family']= 'ptm' #'Times New Roman

    if not os.path.exists("savedPlots/"):
        os.mkdir("savedPlots")

    if system=="Rober":
        
        fact = 1e4
        fig = plt.figure(figsize=(20,10))
        back_colors = ["k","b","darkgreen","darkslategrey"]
        front_colors = ["r","plum","darkturquoise","lightsalmon"]

        for i in range(len(network_sol)):
            if i==1:
                plt.plot(time_plot_sequential,output[:,i]*fact,'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
                plt.plot(time_plot,network_sol[i]*fact,'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
            else:
                plt.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
                plt.plot(time_plot,network_sol[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)

        if avg_time==-1:
            pass #title = f"Comparison of solutions"
        else:
            title = f"Average computational time: {np.round(avg_time,2)}s" if number_iterates>1 \
                            else f"Computational time: {np.round(avg_time,2)}s"
                
            plt.title(title)
        plt.xlabel(r'$t$')
        plt.legend(loc='center', bbox_to_anchor=(1.14,0.5))

        #if avg_time!=-1:
        plt.savefig(f"savedPlots/{system}.pdf",bbox_inches='tight')
        plt.show();

    if system=="SIR":
        
        if len(network_sol_SIR_flow)==0:
            fig = plt.figure(figsize=(30,10))
            
            
            back_colors = ["k","b","darkgreen","darkslategrey"]
            front_colors = ["r","plum","darkturquoise","lightsalmon"]
            
            for i in range(len(network_sol)):
                plt.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
                plt.plot(time_plot,network_sol[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
            
            plt.xlabel(r'$t$')
            
            plt.ylim(bottom=-0.1,top=1.75)
            plt.legend(loc='upper center', ncol=2)
            
            if avg_time==-1:
                pass #title = f"Comparison of solutions,\n RPNN"
            else:
                title = f"RPNN, Average computational time: {np.round(avg_time,2)}s"
                plt.title(title)

            #if avg_time!=-1:
            plt.savefig(f"savedPlots/{system}.pdf",bbox_inches='tight')
            plt.show();
        else:
            fig = plt.figure(figsize=(30,10))
            ax1 = plt.subplot2grid((1, 30), (0, 0), colspan=14)
                        
            back_colors = ["k","b","darkgreen","darkslategrey"]
            front_colors = ["r","plum","darkturquoise","lightsalmon"]
            
            for i in range(len(network_sol)):
                ax1.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
                ax1.plot(time_plot,network_sol[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
            
            if avg_time==-1:
                ax1.set_title(f"RPNN")
            else:
                ax1.set_title(f"RPNN \n Average computational time: {np.round(avg_time,2)}s")
            ax1.set_xlabel(r'$t$')
            
            ax1.set_ylim(bottom=-0.1,top=1.75)
            ax1.legend(loc='upper center', ncol=2)
            
            ax2 = plt.subplot2grid((1, 30), (0, 16), colspan=15)
            
            for i in range(len(network_sol_SIR_flow)):
                ax2.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
                ax2.plot(time_plot,network_sol_SIR_flow[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
            
            ax2.set_xlabel(r'$t$')
            
            ax2.set_ylim(bottom=-0.1,top=1.75)
            ax2.legend(loc='upper center', ncol=2)

            if avg_time==-1:
                pass #ax2.set_title(f"Flow map approach")
            else:
                ax2.set_title(f"Flow map approach \n Average computational time: {np.round(avg_time_SIR_flow,2)}s")
                #fig.suptitle(title)
            plt.subplots_adjust(top=0.76, wspace=0.5)

            #if avg_time!=-1:
            plt.savefig(f"savedPlots/{system}.pdf",bbox_inches='tight')
            plt.show();
    
    if system=="Lorenz":
        
        fig = plt.figure(figsize=(30,10))
        ax1 = plt.subplot2grid((1, 30), (0, 0), colspan=14)
                
        back_colors = ["k","b","darkgreen","darkslategrey"]
        front_colors = ["r","plum","darkturquoise","lightsalmon"]
        
        for i in range(len(network_sol)):
            ax1.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
            ax1.plot(time_plot,network_sol[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
        
        if avg_time==-1:
            ax1.set_title(f"Uniform nodes")
        else:
            ax1.set_title(f"Uniform nodes \n Average computational time: {np.round(avg_time,2)}s")
        ax1.set_xlabel(r'$t$')
        
        ax1.set_ylim(bottom=-30.,top=120.)
        ax1.legend(loc='upper center', ncol=2)
        
        ax2 = plt.subplot2grid((1, 30), (0, 16), colspan=15)
        
        for i in range(len(network_sol)):
            ax2.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
            ax2.plot(time_plot,network_sol_lobatto[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
        
        ax2.set_xlabel(r'$t$')
        
        ax2.set_ylim(bottom=-30.,top=120.)
        ax2.legend(loc='upper center', ncol=2)

        if avg_time==-1:
            ax2.set_title(f"Lobatto nodes")
        else:
            ax2.set_title(f"Lobatto nodes \n Average computational time: {np.round(avg_time_lobatto,2)}s")
        plt.subplots_adjust(top=0.76, wspace=0.5)

        #if avg_time!=-1:
        plt.savefig(f"savedPlots/{system}.pdf",bbox_inches='tight')
        plt.show();
        
    if system=="Brusselator":
        
        fig = plt.figure(figsize=(30,10))
        ax1 = plt.subplot2grid((1, 30), (0, 0), colspan=20)
        
        back_colors = ["k","b","darkgreen","darkslategrey"]
        front_colors = ["r","plum","darkturquoise","lightsalmon"]
        
        for i in range(len(network_sol)):
            ax1.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
            ax1.plot(time_plot,network_sol[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
        
        ax1.set_ylim(-2,max(network_sol.reshape(-1))+0.1)
        ax1.set_xlabel(r'$t$')
        
        ax1.legend(loc='lower center', ncol=2)
        
        ax2 = plt.subplot2grid((1, 30), (0, 22), colspan=8)
        
        ax2.plot(output[:,0],output[:,1],'k-',label="ref",linewidth=5)
        ax2.plot(network_sol[0],network_sol[1],'r--',label="para",linewidth=5)
        
        ax2.set_xlabel(list_of_labels[0])
        ax2.set_ylabel(list_of_labels[1])
        ax2.legend(loc='lower center', ncol=1)
        ax2.set_ylim(-1.,max(network_sol.reshape(-1))+0.1)
        
        if avg_time==-1:
            pass #title = f"Comparison of solutions"
        else:
            title = f"Average computational time: {np.round(avg_time,2)}s" if number_iterates>1 \
                    else f"Computational time: {np.round(avg_time,2)}s"
            fig.suptitle(title)
        
        plt.subplots_adjust(top=0.9, wspace=0.5)

        #if avg_time!=-1:
        plt.savefig(f"savedPlots/{system}.pdf",bbox_inches='tight')
        plt.show();
        
    if system=="Arenstorf":
        
        fig = plt.figure(figsize=(30,10))
        ax1 = plt.subplot2grid((1, 30), (0, 0), colspan=20)
        
        back_colors = ["k","b","darkgreen","darkslategrey"]
        front_colors = ["r","plum","darkturquoise","lightsalmon"]
        
        for i in range(len(network_sol)):
            ax1.plot(time_plot_sequential,output[:,i],'-',color=back_colors[i],label=f"{list_of_labels[i]} ref",linewidth=5)
            ax1.plot(time_plot,network_sol[i],'--',color=front_colors[i],label=f"{list_of_labels[i]} para",linewidth=5)
        
        ax1.set_ylim(-4.5,1.5)
        ax1.set_xlabel(r'$t$')
        
        ax1.legend(loc='lower center', ncol=3)
        
        ax2 = plt.subplot2grid((1, 30), (0, 22), colspan=8)
        
        ax2.plot(output[:,0],output[:,2],'k-',label="ref",linewidth=5)
        ax2.plot(network_sol[0],network_sol[2],'r--',label="para",linewidth=5)
        
        ax2.set_xlabel(list_of_labels[0])
        ax2.set_ylabel(list_of_labels[2])
        ax2.legend(loc='lower center', ncol=1)
        ax2.set_ylim(-2.5,1.5)
        if avg_time==-1:
            pass #title = f"Comparison of solutions"
        else:
            title = f"Average computational time: {np.round(avg_time,2)}s" if number_iterates>1 \
                    else f"Computational time: {np.round(avg_time,2)}s"
        
            fig.suptitle(title)
        plt.subplots_adjust(top=0.9, wspace=0.5)

        #if avg_time!=-1:
        plt.savefig(f"savedPlots/{system}.pdf",bbox_inches='tight')
        plt.show();

    if system=="Burger":

        '''fig = plt.figure(figsize=(40, 13))
        ax1 = plt.subplot2grid((1, 40), (0, 0), colspan=20)

        step = len(time_plot) // 10
        for i in range(0, len(time_plot), step):
            if i == 0:
                ax1.plot(vec.x, output[i], 'k-', linewidth=5, label="ref")
                ax1.plot(vec.x, network_sol[:, i], 'r--', linewidth=5, label="para")
            else:
                ax1.plot(vec.x, output[i], 'k-', linewidth=5)
                ax1.plot(vec.x, network_sol[:, i], 'r--', linewidth=5)

        ax1.set_title("Solution \n snapshots")#,fontsize=45)
        ax1.set_xlabel(r'$x$')#,fontsize=45)
        ax1.set_ylabel(r'$u(x,t)$')#,fontsize=45)
        ax1.legend(loc='upper right', frameon=True, shadow=True)

        ax2 = plt.subplot2grid((1, 40), (0, 20), colspan=9, projection='3d')  # 1 row, 2 columns, second subplot
        ax3 = plt.subplot2grid((1, 40), (0, 30), colspan=9, projection='3d')  # 1 row, 2 columns, second subplot
        

        T, X = np.meshgrid(time_plot, vec.x)
        ax2.plot_surface(X, T, output.T, cmap='viridis', edgecolor='none')
        ax3.plot_surface(X, T, network_sol, cmap='viridis', edgecolor='none')
        ax2.set_title("Reference \n solution",fontsize=45)
        ax3.set_title("Network \n solution",fontsize=45)
        ax2.set_xlabel(r'$x$',fontsize=45)
        ax2.set_ylabel(r'$t$',fontsize=45)
        ax3.set_xlabel(r'$x$',fontsize=45)
        ax3.set_ylabel(r'$t$',fontsize=45)
        
        ax2.tick_params(axis='x', labelrotation=55)
        ax2.set_xticks([0.0,1.0])
        ax2.tick_params(axis='y', labelrotation=-25)
        ax2.set_yticks([0.0,1.0])
        ax2.tick_params(axis='z', pad=20)
        
        ax3.tick_params(axis='x', labelrotation=55)
        ax3.set_xticks([0.0,1.0])
        ax3.tick_params(axis='y', labelrotation=-25)
        ax3.set_yticks([0.0,1.0])
        ax3.tick_params(axis='z', pad=20)

        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if avg_time==-1:
            pass #fig.suptitle(f"Comparison of solutions")
        else:
            fig.suptitle(f"Average computational time: {np.round(avg_time,2)}s" if number_iterates > 1 \
                        else f"Computational time: {np.round(avg_time,2)}s")#,fontsize=45)

        plt.subplots_adjust(top=0.8, wspace=0.5)

        #if avg_time!=-1:
        plt.savefig(f"savedPlots/{system}_{btype}.pdf", bbox_inches='tight')
        plt.show()'''
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FormatStrFormatter

        fig = plt.figure(figsize=(40, 13))
        ax1 = plt.subplot2grid((1, 40), (0, 0), colspan=20)

        step = len(time_plot) // 10
        for i in range(0, len(time_plot), step):
            if i == 0:
                ax1.plot(vec.x, output[i], 'k-', linewidth=5, label="ref")
                ax1.plot(vec.x, network_sol[:, i], 'r--', linewidth=5, label="para")
            else:
                ax1.plot(vec.x, output[i], 'k-', linewidth=5)
                ax1.plot(vec.x, network_sol[:, i], 'r--', linewidth=5)

        ax1.set_title("Solution \n snapshots")#,fontsize=45)
        ax1.set_xlabel(r'$x$')#,fontsize=45)
        ax1.set_ylabel(r'$u(x,t)$')#,fontsize=45)
        ax1.legend(loc='upper right', frameon=True, shadow=True)

        ax2 = plt.subplot2grid((1, 40), (0, 20), colspan=9, projection='3d')  # 1 row, 2 columns, second subplot
        ax3 = plt.subplot2grid((1, 40), (0, 30), colspan=9, projection='3d')  # 1 row, 2 columns, second subplot

        T, X = np.meshgrid(time_plot, vec.x)
        ax2.plot_surface(X, T, output.T, cmap='viridis', edgecolor='none')
        ax3.plot_surface(X, T, network_sol, cmap='viridis', edgecolor='none')
        ax2.set_title("Reference \n solution",fontsize=45)
        ax3.set_title("Network \n solution",fontsize=45)
        ax2.set_xlabel(r'$x$',fontsize=45)
        ax2.set_ylabel(r'$t$',fontsize=45)
        ax3.set_xlabel(r'$x$',fontsize=45)
        ax3.set_ylabel(r'$t$',fontsize=45)

        ax2.tick_params(axis='x', labelrotation=55)
        ax2.set_xticks([0.0,1.0])
        ax2.tick_params(axis='y', labelrotation=-25)
        ax2.set_yticks([0.0,1.0])
        ax2.tick_params(axis='z', pad=20)

        ax3.tick_params(axis='x', labelrotation=55)
        ax3.set_xticks([0.0,1.0])
        ax3.tick_params(axis='y', labelrotation=-25)
        ax3.set_yticks([0.0,1.0])
        ax3.tick_params(axis='z', pad=20)

        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # Adding annotations (a) and (b)
        fig.text(0.3, -0.05, '(a)', fontweight='bold', ha='center', transform=fig.transFigure)
        fig.text(0.7, -0.05, '(b)', fontweight='bold', ha='center', transform=fig.transFigure)

        if avg_time==-1:
            pass #fig.suptitle(f"Comparison of solutions")
        else:
            fig.suptitle(f"Average computational time: {np.round(avg_time,2)}s" if number_iterates > 1 \
                        else f"Computational time: {np.round(avg_time,2)}s")#,fontsize=45)

        plt.subplots_adjust(top=0.8, wspace=0.5)

        #if avg_time!=-1:
        plt.savefig(f"savedPlots/{system}_{btype}.pdf", bbox_inches='tight')
        plt.show()

