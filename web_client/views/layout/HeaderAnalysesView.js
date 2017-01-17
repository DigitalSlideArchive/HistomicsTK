import _ from 'underscore';
import { restRequest } from 'girder/rest';

import router from '../../router';
import View from '../View';
import headerAnalysesTemplate from '../../templates/layout/headerAnalyses.pug';
import '../../stylesheets/layout/headerAnalyses.styl';

import 'bootstrap-submenu/dist/js/bootstrap-submenu';
import 'bootstrap-submenu/dist/css/bootstrap-submenu.css';

var HeaderUserView = View.extend({
    events: {
        'click .h-analysis-item': '_setAnalysis'
    },
    render() {
        restRequest({
            path: 'HistomicsTK/HistomicsTK/docker_image'
        }).then((analyses) => {
            if (_.keys(analyses || {}).length > 0) {
                this.$el.html(headerAnalysesTemplate({
                    analyses: analyses || {}
                }));
                this.$('.h-analyses-dropdown-link').submenupicker();
            }
        });
        return this;
    },
    _setAnalysis(evt) {
        evt.preventDefault();
        var target = $(evt.currentTarget).data();

        router.setQuery('analysis', target.api, {trigger: true});
    }
});

export default HeaderUserView;
