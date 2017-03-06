import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import router from '../router';

import regionSelector from '../templates/panels/regionSelector.pug';
import '../stylesheets/panels/regionSelector.styl';

var RegionSelector = Panel.extend({
    events: {
        'click .h-select-region-button': 'selectRegion',
        'click .h-clear-region-button': 'clearRegion'
    },
    render() {
        this.$el.html(regionSelector({
            id: 'region-panel-container',
            title: 'Region'
        }));
    },
    selectRegion() {
        var bounds = router.getQuery('bounds').split(',');
        var left = parseFloat(bounds[0]);
        var top = parseFloat(bounds[1]);
        var right = parseFloat(bounds[2]);
        var bottom = parseFloat(bounds[3]);
        this.$('.h-region-value').val(JSON.stringify({
            left, right, top, bottom
        }));
    },
    clearRegion() {
        this.$('.h-region-value').val('');
    }
});

export default RegionSelector;
