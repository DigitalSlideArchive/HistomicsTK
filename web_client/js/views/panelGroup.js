histomicstk.views.PanelGroup = girder.View.extend({
    events: {
        'click .h-info-panel-reload': 'reload',
        'click .h-info-panel-submit': 'submit',
        'click .h-remove-panel': 'removePanel'
    },
    initialize: function () {
        this.panels = [];
        this._panelViews = {};
        this._schemaName = null;

        // render a specific schema
        this.listenTo(histomicstk.router, 'route:gui', this.schema);

        // remove the current schema reseting to the default view
        this.listenTo(histomicstk.router, 'route:main', this.reset);
    },
    render: function () {
        this.$el.html(histomicstk.templates.panelGroup({
            info: this._gui,
            panels: this.panels
        }));
        _.each(this._panelViews, function (view) {
            view.remove();
        });
        _.each(this.panels, _.bind(function (panel) {
            this._panelViews[panel.id] = new histomicstk.views.ControlsPanel({
                parentView: this,
                collection: new histomicstk.collections.Widget(panel.parameters),
                title: panel.label,
                advanced: panel.advanced,
                el: this.$el.find('#' + panel.id)
            });

            this._panelViews[panel.id].render();
        }, this));
    },

    /**
     * Submit the current values to the server.
     */
    submit: function () {
        // todo
        console.log('Submit ' + this._schemaName); // eslint-disable-line no-console
        console.log(JSON.stringify(this.parameters(), null, 2)); // eslint-disable-line no-console
    },

    /**
     * Get the current values of all of the parameters contained in the gui.
     * Returns an object that maps each parameter id to it's value.
     */
    parameters: function () {
        return _.chain(this._panelViews)
            .pluck('collection')
            .invoke('values')
            .reduce(function (a, b) {
                return _.extend(a, b);
            }, {})
            .value();
    },

    /**
     * Remove a panel after confirmation from the user.
     */
    removePanel: function (e) {
        girder.confirm({
            text: 'Are you sure you want to remove this panel?',
            confirmCallback: _.bind(function () {
                var el = $(e.currentTarget).data('target');
                var id = $(el).attr('id');
                this.panels = _.filter(this.panels, function (panel) {
                    return panel.id !== id;
                });
                this.render();
            }, this)
        });
    },

    /**
     * Remove all panels.
     */
    reset: function () {
        this._schemaName = null;
        this.panels = [];
        this._gui = null;
        this.render();
    },

    /**
     * Restore all panels to the default state.
     */
    reload: function () {
        if (!this._gui) {
            return this;
        }

        // Create a panel for each "group" in the schema, and copy
        // the advanced property from the parent panel.
        this.panels = _.chain(this._gui.panels).map(function (panel) {
            return _.map(panel.groups, function (group) {
                group.advanced = !!panel.advanced;
                group.id = _.uniqueId('panel-');
                return group;
            });
        }).flatten(true).value();

        this.render();
        return this;
    },

    /**
     * Generate panels from a slicer XML schema stored on the server.
     */
    schema: function (s) {

        girder.restRequest({
            path: '/HistomicsTK/' + s + '/xmlspec'
        }).then(_.bind(function (xml) {
            var fail = !xml;
            try {
                this._gui = histomicstk.schema.parse(xml);
            } catch (e) {
                fail = true;
            }

            if (fail) {
                girder.events.trigger('g:alert', {
                    icon: 'attention',
                    text: 'Invalid XML schema',
                    type: 'danger'
                });
                histomicstk.router.navigate('', {trigger: true});
                this.reset();
                return this;
            }

            this._schemaName = s;
            this.reload();

            return this;
        }, this));

        return this;
    }
});
