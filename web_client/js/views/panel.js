histomicstk.views.Panel = girder.View.extend({
    events: {
        'click .h-remove-panel': 'askRemove',
        'show.bs.collapse': 'expand',
        'hide.bs.collapse': 'collapse'
    },

    askRemove: function () {
        girder.confirm({
            text: 'Are you sure you want to remove this panel?',
            confirmCallback: _.bind(this.remove, this)
        });
    }
});
