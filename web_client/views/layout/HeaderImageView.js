import events from '../../events';
import View from '../View';

import headerImageTemplate from '../../templates/layout/headerImage.pug';
import '../../stylesheets/layout/headerImage.styl';

var HeaderImageView = View.extend({
    events: {
        'click .h-open-image': function (evt) {
            events.trigger('h:openImageUi');
        },
        'click .h-open-annotated-image': function (evt) {
            events.trigger('h:openAnnotatedImageUi');
        }
    },

    initialize() {
        this.imageModel = null;
        this.listenTo(events, 'h:imageOpened', (model) => {
            this.imageModel = model;
            this.render();
        });
    },

    render() {
        this.$el.html(headerImageTemplate({
            image: this.imageModel
        }));
        return this;
    }
});

export default HeaderImageView;
