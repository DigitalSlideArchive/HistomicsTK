histomicstk.views.Visualization = girder.View.extend({
    initialize: function () {
        this._map = geo.map({
            node: '<div style="width: 100%; height: 100%"/>',
            width: 100,
            height: 100
        });
        this._map.createLayer('osm', {
            url: function () { return 'http://lorempixel.com/256/256/cats/'; },
            keepLower: false,
            wrapX: true,
            wrapY: true,
            attribution: 'Images provided by <a href="http://lorempixel.com">lorempixel</a>'
        });
    },
    render: function () {
        var mapnode = this._map.node();
        if (this.$el) {
            this.$el.empty();
            this.$el.append(mapnode);
            this._map.size({
                width: this.$el.width(),
                height: this.$el.height()
            });
        }
        return this;
    },
    destroy: function () {
        this._map.exit();
        girder.View.prototype.destroy.apply(this, arguments);
    }
});
