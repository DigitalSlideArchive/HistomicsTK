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
        this._schema = null;
        this.panels = [];
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
        // replace with an appropriate call to the server
        var schema = this.testSchemas[s];

        var fail = !schema;
        try {
            this._gui = histomicstk.schema.parse(this.testSchemas[s]);
        } catch (e) {
            fail = true;
        }

        this._schemaName = s;
        if (fail) {
            girder.events.trigger('g:alert', {
                icon: 'attention',
                text: 'Invalid XML schema',
                type: 'danger'
            });
            histomicstk.router.navigate('', {trigger: true});
            this._gui = null;
            return this;
        }

        this.reload();
        return this;
    },

    testSchemas: {
        a: [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<executable>',
            '<category>Tours</category>',
            '<title>Execution Model Tour</title>',
            '<description>',
            'Shows one of each type of parameter.',
            '</description>',
            '<version>1.0</version>',
            '<documentation-url></documentation-url>',
            '<license></license>',
            '<contributor>Daniel Blezek</contributor>',
            '<parameters>',
            '<label>Scalar Parameters</label>',
            '<description>',
            'Variations on scalar parameters',
            '</description>',
            '<integer>',
            '<name>integerVariable</name>',
            '<flag>i</flag>',
            '<longflag>integer</longflag>',
            '<description>',
            'An integer without constraints',
            '</description>',
            '<label>Integer Parameter</label>',
            '<default>30</default>',
            '</integer>',
            '<label>Scalar Parameters With Constraints</label>',
            '<description>Variations on scalar parameters</description>',
            '<double>',
            '<name>doubleVariable</name>',
            '<flag>d</flag>',
            '<longflag>double</longflag>',
            '<description>An double with constraints</description>',
            '<label>Double Parameter</label>',
            '<default>30</default>',
            '<constraints>',
            '<minimum>0</minimum>',
            '<maximum>1.e3</maximum>',
            '<step>0</step>',
            '</constraints>',
            '</double>',
            '</parameters>',
            '<parameters advanced="true">',
            '<label>Vector Parameters</label>',
            '<description>Variations on vector parameters</description>',
            '<float-vector>',
            '<name>floatVector</name>',
            '<flag>f</flag>',
            '<description>A vector of floats</description>',
            '<label>Float Vector Parameter</label>',
            '<default>1.3,2,-14</default>',
            '</float-vector>',
            '<string-vector>',
            '<name>stringVector</name>',
            '<longflag>string_vector</longflag>',
            '<description>A vector of strings</description>',
            '<label>String Vector Parameter</label>',
            '<default>"foo",bar,"foobar"</default>',
            '</string-vector>',
            '</parameters>',
            '<parameters>',
            '<label>Enumeration Parameters</label>',
            '<description>Variations on enumeration parameters</description>',
            '<string-enumeration>',
            '<name>stringChoice</name>',
            '<flag>e</flag>',
            '<longflag>enumeration</longflag>',
            '<description>An enumeration of strings</description>',
            '<label>String Enumeration Parameter</label>',
            '<default>foo</default>',
            '<element>foo</element>',
            '<element>"foobar"</element>',
            '<element>foofoo</element>',
            '</string-enumeration>',
            '</parameters>',
            '</executable>'
        ].join('')
    }
});
