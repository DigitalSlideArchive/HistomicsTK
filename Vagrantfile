Vagrant.configure("2") do |config|
  config.vm.box = "bento/ubuntu-16.04"


  if Vagrant.has_plugin?("vagrant-cachier")
    config.cache.scope = :box
    config.cache.enable :apt
    config.cache.enable :yum
    config.cache.enable :npm
  end

  # The exposed ports can be changed here; the ssh port is never necessary.
  config.vm.network "forwarded_port", guest: 22, host: 2209
  config.vm.network "forwarded_port", guest: 8080, host: 8009
  config.vm.provider "virtualbox" do |v|
    v.name = "HistomicsTK Ubuntu 16.04"
    # You may need to configure this to run benignly on your host machine
    v.memory = 4096
    v.cpus = 4
    # Size the disk to a specific number of Mbytes.
    # v.customize ["modifyhd", ":id", "--resize", 100 * 1024]
  end

  provisioner_type = if
      Gem::Version.new(Vagrant::VERSION) > Gem::Version.new('1.8.1')
    then
      # Vagrant > 1.8.1 is required due to
      # https://github.com/mitchellh/vagrant/issues/6793
      "ansible_local"
    else
      "ansible"
    end
  config.vm.provision provisioner_type do |ansible|
    ansible.playbook = "ansible/vagrant.yml"
    ansible.galaxy_role_file = "ansible/requirements.yml"

    if provisioner_type == "ansible_local"
      ansible.provisioning_path = "/vagrant"
      ansible.install_mode = "pip"
      ansible.version = "2.4.4.0"
    end
  end
end
