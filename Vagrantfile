Vagrant.configure("2") do |config|
  config.vm.box_url = "https://cloud-images.ubuntu.com/vagrant/trusty/current/trusty-server-cloudimg-amd64-vagrant-disk1.box"
  config.vm.box = "ubuntu/trusty_64"
  # The exposed ports can be changed here; the ssh port is never necessary.
  config.vm.network "forwarded_port", guest: 22, host: 2209
  config.vm.network "forwarded_port", guest: 8009, host: 8009
  config.vm.provider "virtualbox" do |v|
    v.name = "HistomicsTK Ubuntu 14.04"
    # You may need to configure this to run benignly on your host machine
    v.memory = 4096
    v.cpus = 4
    # Size the disk to a specific number of Mbytes.
    # v.customize ["modifyhd", ":id", "--resize", 100 * 1024]
  end

  config.vm.provision "ansible" do |ansible|
    ansible.playbook = "ansible/vagrant.yml"
    ansible.verbose = "v"
  end
end
